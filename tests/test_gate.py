"""reflexive.gate.DevGate: the deployment rule agreed in SEL-51, check by check.

Records are built by hand, and ``saved`` stands in for the oracle (no τ²-bench, no API): it
gives AuditKIT's verdicts from each record's saved outcome. The hallucination counts come from
the real classifier.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from reflexive import splits
from reflexive.gate import NO_DECISION, PROMOTE, REJECT, DevGate
from reflexive.metrics import is_broken

Record = dict[str, Any]

DEV = [(d, t) for d in splits.DOMAINS for t in splits.task_ids(d, "dev")]  # the 36 dev tasks
SYS = {"role": "system", "content": "policy"}
HI = {"role": "user", "content": "I need help"}
DONE = {"role": "assistant", "content": "Done."}


def wrong_tool(i: int) -> Record:
    """The agent calls a tool it doesn't have: one TOOL finding."""
    call = {"id": f"c{i}", "type": "function", "function": {"name": "no_such_tool", "arguments": "{}"}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def rec(task: tuple[str, str], ok: bool, *, tools: int = 1, **kw: Any) -> Record:
    """A conversation that passed (no findings) or failed with ``tools`` TOOL findings."""
    view = [SYS, HI, DONE] if ok else [SYS, HI, *[wrong_tool(i) for i in range(tools)]]
    r: Record = {
        "domain": task[0],
        "task_id": task[1],
        "termination": "user_stop",
        "outcome": {"passed": ok, "skipped": []},
        "messages": view,
        "agent_view": view,
        "error": None,
    }
    r.update(kw)
    return r


def run(passes: set[tuple[str, str]], **per_task: dict[tuple[str, str], dict[str, Any]]) -> list[Record]:
    """One adapter's dev run: the tasks in ``passes`` pass; ``per_task`` overrides some records."""
    over = per_task.get("over", {})
    return [rec(t, t in passes, **over.get(t, {})) for t in DEV]


def saved(rec: Record) -> SimpleNamespace:
    """The oracle's verdict, as the oracle would give it, from the record's saved outcome."""
    o = rec.get("outcome") or {}
    if is_broken(rec):
        return SimpleNamespace(verdict="error", reason="the run broke")
    if o.get("passed") and o.get("skipped"):
        return SimpleNamespace(
            verdict="unknown", reason=f"{', '.join(o['skipped'])} were skipped (judges off)"
        )
    return SimpleNamespace(verdict="success" if o.get("passed") else "failure", reason="saved")


SAVED_ID = {"type": "test", "version": "saved outcomes"}
saved.identity = SAVED_ID  # ty: ignore[unresolved-attribute]


def gate(**kw: Any) -> DevGate:
    return DevGate.load(score=saved, **kw)


BASE = set(DEV[:2])  # the current adapter passes 2 of 36, like Arm A on dev


# --- The agreed thresholds ----------------------------------------------------------------------


def test_the_config_holds_the_agreed_rule() -> None:
    g = DevGate.load()
    assert (g.margin, g.max_crashed, g.samples_per_task, g.domain_loss, g.max_cycles) == (2, 2, 1, 2, 3)
    assert g.finding_types == ("TOOL", "ARG", "LOOP")  # Loop counts (SEL-51, 6 Nov)
    assert g.findings_tolerance == 0.10 and g.score is None  # None: the oracle, built on first use
    assert g.min_tasks == len(DEV) == 36  # every dev task, in configs/splits.json


def test_an_unknown_setting_or_finding_type_is_refused(tmp_path) -> None:
    f = tmp_path / "gate.yaml"
    f.write_text("margin: 2\nmargn: 3\n")
    with pytest.raises(ValueError, match="unknown gate settings.*margn"):
        DevGate.load(f)
    with pytest.raises(ValueError):
        gate(finding_types=["TOOL", "ARGS"])
    f.write_text("score: x\n")  # the scorer isn't a setting
    with pytest.raises(ValueError, match="unknown gate settings.*score"):
        DevGate.load(f)


# --- 2. Pass rate -------------------------------------------------------------------------------


def test_two_more_tasks_promotes_and_records_the_statistics() -> None:
    d = gate().decide(run(BASE), run(BASE | set(DEV[2:4])))
    assert d.verdict == PROMOTE and d.passed and d.check == "all"
    assert d.details["net_gain"] == 2 and d.details["passed"] == {"baseline": 2, "candidate": 4}
    assert 0 < d.details["mcnemar_p"] <= 1 and len(d.details["bootstrap95_gain"]) == 2
    assert "not required" in d.reason


def test_one_more_task_is_rejected() -> None:
    d = gate().decide(run(BASE), run(BASE | {DEV[2]}))
    assert d.verdict == REJECT and not d.passed and d.check == "pass_rate" and "needs +2" in d.reason


def test_the_gain_is_net_of_tasks_the_candidate_lost() -> None:
    gains = set(DEV[2:5])  # 3 new passes
    assert gate().decide(run(BASE), run({DEV[1]} | gains)).verdict == PROMOTE  # lost 1: net +2
    d = gate().decide(run(BASE), run(gains))  # lost both: net +1
    assert d.verdict == REJECT and d.details["net_gain"] == 1


# --- 3. Hallucination ---------------------------------------------------------------------------


def test_more_findings_per_conversation_is_rejected_even_with_the_gain() -> None:
    worse = {t: {"tools": 3} for t in DEV[10:20]}  # these failures now call three unknown tools
    d = gate().decide(run(BASE), run(BASE | set(DEV[2:4]), over=worse))
    assert d.verdict == REJECT and d.check == "hallucination" and "10% tolerance" in d.reason
    f = d.details["findings_per_conversation"]
    assert f["candidate"]["TOOL"] > f["baseline"]["TOOL"]


def test_a_rise_within_the_tolerance_is_allowed() -> None:
    """A small rise in findings per conversation passes; a large one, or any with tolerance 0, doesn't.

    Same passes on both sides (margin 0), so only the findings differ."""
    three = {t: {"tools": 3} for t in DEV}
    base = run(BASE, over=three)
    small = run(BASE, over=three | {DEV[4]: {"tools": 6}})  # one conversation got worse
    large = run(BASE, over=three | {t: {"tools": 6} for t in DEV[4:20]})  # sixteen did

    d = gate(margin=0).decide(base, small)
    f = d.details["findings_per_conversation"]
    fb, fc = f["baseline"]["total"], f["candidate"]["total"]
    assert fb < fc <= fb * 1.10 and d.verdict == PROMOTE and f"up to {fb * 1.1:.2f}" in d.reason
    d = gate(margin=0).decide(base, large)
    f = d.details["findings_per_conversation"]
    assert f["candidate"]["total"] > f["baseline"]["total"] * 1.10 and d.check == "hallucination"
    assert gate(margin=0, findings_tolerance=0).decide(base, small).check == "hallucination"


def test_loop_findings_count_unless_the_config_leaves_them_out() -> None:
    loops = {t: {"termination": "max_steps"} for t in DEV[10:20]}  # step limit: a LOOP finding
    cand = run(BASE | set(DEV[2:4]), over=loops)
    d = gate().decide(run(BASE), cand)
    assert d.check == "hallucination" and d.details["findings_per_conversation"]["candidate"]["LOOP"] > 0
    assert gate(finding_types=["TOOL", "ARG"]).decide(run(BASE), cand).verdict == PROMOTE


# --- 4. Domains ---------------------------------------------------------------------------------


def test_a_domain_losing_two_tasks_is_rejected() -> None:
    retail = [t for t in DEV if t[0] == "retail"]
    telecom = [t for t in DEV if t[0] == "telecom"]
    base = set(telecom[:2])
    d = gate().decide(run(base), run(set(retail[:4])))  # net +2 overall, telecom -2
    assert d.details["net_gain"] == 2 and d.details["domains_net"]["telecom"] == -2
    assert d.verdict == REJECT and d.check == "domains" and "telecom -2" in d.reason


# --- 1. Complete run: no decision, not a rejection -----------------------------------------------


def test_up_to_two_crashes_are_allowed_and_left_out_of_the_pairing() -> None:
    crashed = {t: {"error": "RateLimitError: 429"} for t in DEV[2:4]}
    d = gate().decide(run(BASE), run(BASE | set(DEV[2:6]), over=crashed))
    assert d.verdict == PROMOTE and d.details["scored"] == 34
    assert d.details["dropped"] == [list(t) for t in sorted(DEV[2:4])]  # left out of both sides


@pytest.mark.parametrize("broken", [{"error": "RateLimitError"}, {"termination": "infrastructure_error"}])
def test_three_crashes_is_no_decision(broken: dict[str, Any]) -> None:
    d = gate().decide(run(BASE), run(BASE | set(DEV[5:9]), over={t: broken for t in DEV[2:5]}))
    assert d.verdict == NO_DECISION and d.check == "complete" and "re-run the gate" in d.reason


def test_missing_or_different_tasks_is_no_decision() -> None:
    d = gate().decide(run(BASE), run(BASE)[:-1])
    assert d.verdict == NO_DECISION and "35 tasks, fewer than 36" in d.reason
    # Each adapter left out a different dev task (a made-up or non-dev task is refused outright)
    b, c = run(BASE)[:-1], run(BASE)[:-2] + run(BASE)[-1:]
    assert "different tasks" in gate(min_tasks=len(DEV) - 1).decide(b, c).reason


def test_scores_with_judged_checks_skipped_are_no_decision() -> None:
    skipped = {DEV[0]: {"outcome": {"passed": True, "skipped": ["NL_ASSERTION"]}}}
    d = gate().decide(run(BASE), run(BASE | set(DEV[2:4]), over=skipped))
    assert d.verdict == NO_DECISION and "unverified" in d.reason and "judges off" in d.reason


def test_two_records_for_one_task_is_refused() -> None:
    with pytest.raises(ValueError, match="2 records; the gate takes 1 per task"):
        gate().decide(run(BASE), run(BASE) + [rec(DEV[0], True)])


# --- The budget: 3 heal cycles on the same dev tasks ---------------------------------------------


def test_after_three_decisions_the_gate_says_run_the_test_split() -> None:
    g = gate()
    g.decide(run(BASE), run(BASE)[:-1])  # incomplete: doesn't use up a cycle
    for _ in range(3):
        assert g.decide(run(BASE), run(BASE)).verdict == REJECT
    d = g.decide(run(BASE), run(BASE | set(DEV[2:8])))
    assert d.verdict == NO_DECISION and d.check == "budget" and "run the test split" in d.reason
    assert g.cycles_used == 3 and len(g.history) == 5


# --- Scoring: the oracle, not the record ---------------------------------------------------------


def test_the_verdicts_come_from_the_scorer_not_the_saved_outcome() -> None:
    """The records say nothing passed; the scorer (the oracle in production) says two more did."""
    rescored = BASE | set(DEV[2:4])

    def oracle(rec: Record) -> SimpleNamespace:
        ok = (rec["domain"], rec["task_id"]) in rescored and rec["adapter"] == "new"
        was = (rec["domain"], rec["task_id"]) in BASE
        return SimpleNamespace(verdict="success" if ok or was else "failure", reason="")

    base = [dict(r, adapter="old") for r in run(set())]
    cand = [dict(r, adapter="new") for r in run(set())]
    d = DevGate.load(score=oracle).decide(base, cand)
    assert d.verdict == PROMOTE and d.details["passed"] == {"baseline": 2, "candidate": 4}


def test_by_default_the_gate_scores_with_the_oracle_judges_on(monkeypatch) -> None:
    import reflexive.gate as gate_mod

    built = []

    def fake_oracle_score(judges: bool = True) -> Any:
        built.append(judges)
        return saved

    monkeypatch.setattr(gate_mod, "oracle_score", fake_oracle_score)
    g = DevGate.load()
    g.decide(run(BASE), run(BASE))
    g.decide(run(BASE), run(BASE))
    assert built == [True] and g.score is saved  # built once, on first use
    assert g.history[0].details["scored_by"] == SAVED_ID


def test_the_oracle_scorer_turns_a_record_that_isnt_a_conversation_into_an_error() -> None:
    pytest.importorskip("auditkit")
    from reflexive.gate import oracle_score

    score = oracle_score(judges=False)
    assert score({"domain": "telecom"}).verdict == "error"
    assert score.identity["type"] == "tau2_score"  # ty: ignore[unresolved-attribute]


# --- Review of #46 (7 Nov) -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("split", "message"), [("test", "TEST task"), ("train", "is a train task; the gate takes dev tasks only")]
)
def test_the_gate_refuses_anything_but_dev_tasks(split: str, message: str) -> None:
    """Another split's tasks are refused, even with a clear gain: gating on test spends the final eval."""
    tasks = [(d, t) for d in splits.DOMAINS for t in splits.task_ids(d, split)][: len(DEV)]
    base = [rec(t, False) for t in tasks]
    cand = [rec(t, i < 4) for i, t in enumerate(tasks)]
    g = gate()
    with pytest.raises(ValueError, match=message):
        g.decide(base, cand)
    assert g.history == []  # nothing decided, no cycle used


def test_a_task_in_no_split_is_refused() -> None:
    odd = [*run(BASE)[:-1], rec(("retail", "no-such-task"), False)]
    with pytest.raises(ValueError, match="in no split"):
        gate().decide(odd, odd)


def test_the_budget_is_kept_across_gates_through_the_history_file(tmp_path) -> None:
    """Heal cycles run as separate jobs: a new gate reading the same file knows the cycles used."""
    path = tmp_path / "gate" / "history.jsonl"
    first = gate(history_file=path)
    first.decide(run(BASE), run(BASE)[:-1])  # no decision: doesn't use a cycle
    for _ in range(2):
        first.decide(run(BASE), run(BASE))  # two rejects
    later = gate(history_file=path)  # e.g. the next day, in a new process
    assert later.cycles_used == 2 and [d.verdict for d in later.history] == [NO_DECISION, REJECT, REJECT]
    assert later.decide(run(BASE), run(BASE)).verdict == REJECT  # the third cycle
    d = gate(history_file=path).decide(run(BASE), run(BASE | set(DEV[2:8])))
    assert d.verdict == NO_DECISION and d.check == "budget"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 5 and all("decided_at" in r and "verdict" in r for r in rows)


def test_without_a_history_file_a_new_gate_starts_fresh() -> None:
    g = gate()
    g.decide(run(BASE), run(BASE))
    assert g.cycles_used == 1 and gate().cycles_used == 0


def test_one_extra_finding_over_a_clean_baseline_isnt_a_rejection() -> None:
    """A relative tolerance on 0 findings would reject a single finding; one is always allowed."""
    clean = [rec(t, True) for t in DEV]  # all pass, 0 findings
    better = [rec(t, True) for t in DEV]
    better[5] = rec(DEV[5], True, agent_view=[*better[5]["agent_view"][:2], wrong_tool(0), DONE])
    g = gate(margin=0)
    assert g.decide(clean, better).verdict == PROMOTE
    worse = [rec(t, True) for t in DEV]
    for i in (5, 6):
        worse[i] = rec(DEV[i], True, agent_view=[*worse[i]["agent_view"][:2], wrong_tool(0), DONE])
    d = gate(margin=0).decide(clean, worse)
    assert d.verdict == REJECT and d.check == "hallucination"


def test_a_misspelt_keyword_override_is_refused_like_a_misspelt_setting() -> None:
    with pytest.raises(ValueError, match="unknown gate settings passed to DevGate.load: margn"):
        DevGate.load(margn=2)
