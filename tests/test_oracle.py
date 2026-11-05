"""reflexive.oracle: τ²-bench's scoring as an AuditKIT outcome oracle (SEL-46).

The first tests are offline. The rest replay each task's correct actions through τ²-bench's
evaluator (no model, no API calls; judges off) and are skipped without τ²-bench.
"""

from __future__ import annotations

from typing import Any

import pytest
from auditkit.agent_eval.outcome import ERROR, FAILURE, SUCCESS, UNKNOWN, resolve_outcome
from test_golden import record

from reflexive.oracle import Tau2Oracle, case_for, episode_from_record

Record = dict[str, Any]


def verdict(rec: Record, *, judges: bool = False) -> Any:
    oracle = Tau2Oracle(judges=judges)
    return resolve_outcome(episode_from_record(rec), case_for(rec, oracle))  # as AuditKIT's runner calls it


# --- Offline ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "broken",
    [{"error": "RateLimitError: 429 too many requests"}, {"termination": "infrastructure_error"}],
)
def test_a_broken_run_is_an_error_never_a_failure(broken: Record) -> None:
    out = verdict(record(**broken))
    assert out.verdict == ERROR and "not a model failure" in out.reason


def test_an_episode_without_its_record_is_an_error() -> None:
    oracle = Tau2Oracle()
    assert oracle.evaluate(object()).verdict == ERROR


def test_the_identity_names_the_scorer_and_whether_judges_ran() -> None:
    on, off = Tau2Oracle(judges=True).identity(), Tau2Oracle(judges=False).identity()
    assert on["type"] == "custom_predicate" and on["fn"] == "reflexive.outcome.score"
    assert "tau2-bench 2.0.0" in on["version"] and on["version"] != off["version"]
    assert "judges off" in off["version"]


def test_the_episode_shows_what_the_agent_saw_and_keeps_the_record_for_scoring() -> None:
    rec = record(outcome={"passed": True, "reward": 1.0, "checks": {}, "skipped": []})
    ep = episode_from_record(rec)
    calls = [e for e in ep.events if e.type == "tool_call"]
    assert calls and all("check_status_bar" not in str(e.payload) for e in calls)  # the customer's own action
    assert ep.metadata["tau2_record"] is rec and ep.stop_reason == rec["termination"]
    assert ep.source_verdict == SUCCESS  # provenance only: AuditKIT never acts on it
    assert episode_from_record(record(error="boom")).source_verdict is None


# --- τ²-bench's evaluator ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def retail() -> Any:
    pytest.importorskip("tau2")
    from reflexive.outcome import scorer

    return scorer("retail")


def replay(sc: Any, task: str, *, drop: int | None = None, termination: str = "user_stop") -> Record:
    """A record whose conversation is the task's correct actions (optionally minus one)."""
    return {
        "domain": sc.domain,
        "task_id": task,
        "termination": termination,
        "messages": sc.gold_trajectory(task, drop=drop),
        "outcome": {"passed": None},
        "error": None,
    }


def _task(sc: Any, *, judged: bool) -> str:
    """The first retail task that changes the database, with or without LLM-judged checks."""
    for t, task in sc.tasks.items():
        basis = (
            {b.value for b in task.evaluation_criteria.reward_basis} if task.evaluation_criteria else set()
        )
        if sc.write_action_indices(t) and ("NL_ASSERTION" in basis) == judged and "DB" in basis:
            return t
    pytest.skip(f"no retail task with judged={judged}")


def test_the_tasks_correct_actions_succeed(retail: Any) -> None:
    out = verdict(replay(retail, _task(retail, judged=False)))
    assert out.verdict == SUCCESS and out.evidence["reward"] == 1.0
    assert out.source and out.source["fn"] == "reflexive.outcome.score"


def test_leaving_out_the_last_database_change_fails(retail: Any) -> None:
    t = _task(retail, judged=False)
    out = verdict(replay(retail, t, drop=retail.write_action_indices(t)[-1]))
    assert out.verdict == FAILURE and out.evidence["checks"]["DB"] == 0.0


def test_a_conversation_cut_short_fails_even_with_the_correct_actions(retail: Any) -> None:
    out = verdict(replay(retail, _task(retail, judged=False), termination="max_steps"))
    assert out.verdict == FAILURE and "prematurely" in out.reason


def test_with_judges_off_a_deterministic_pass_is_unknown_not_success(retail: Any) -> None:
    """The judged checks weren't run, so the pass isn't verified (AuditKIT's AG-07)."""
    out = verdict(replay(retail, _task(retail, judged=True)), judges=False)
    assert out.verdict == UNKNOWN and "NL_ASSERTION" in out.reason


def test_with_judges_off_a_failed_database_check_is_still_a_failure(retail: Any) -> None:
    t = _task(retail, judged=True)
    out = verdict(replay(retail, t, drop=retail.write_action_indices(t)[-1]), judges=False)
    assert out.verdict == FAILURE


# --- just oracle-score's summary -------------------------------------------------------------


def test_the_summary_rates_only_scored_conversations_and_reports_disagreements() -> None:
    from types import SimpleNamespace as NS

    from _scripts import load_script

    os_ = load_script("arms/oracle_score")
    r = {"domain": "retail", "task_id": "1"}
    scored = [
        (r, NS(verdict=SUCCESS, reason="passed"), SUCCESS),
        (r, NS(verdict=FAILURE, reason="DB"), SUCCESS),  # the oracle disagrees with what the run saved
        (r, NS(verdict=UNKNOWN, reason="judges off"), SUCCESS),  # not counted either way
        (r, NS(verdict=ERROR, reason="crash"), None),
    ]
    s = os_.summary(scored)
    assert s["verdicts"] == {"success": 1, "failure": 1, "unknown": 1, "error": 1}
    assert s["success_rate"] is None  # an unknown could go either way: no single rate
    assert s["success_rate_bounds"] == [round(1 / 3, 4), round(2 / 3, 4)]
    assert "between 33.3% and 66.7%" in os_.rate_text(s)
    assert [d["oracle"] for d in s["disagreements_with_saved"]] == [FAILURE]
    assert os_.summary(scored[:2])["success_rate"] == 0.5  # no unknowns: a plain rate


# --- From the review of #43 ------------------------------------------------------------------


def test_with_judges_off_an_early_ending_is_a_failure_not_unknown(retail: Any) -> None:
    """τ²-bench scores an early ending 0 with no checks run: nothing passed (retail dev 35 and 73)."""
    out = verdict(replay(retail, _task(retail, judged=True), termination="too_many_errors"), judges=False)
    assert out.verdict == FAILURE and out.evidence["checks"] == {}


def test_a_record_with_no_ending_is_an_error_not_scored_as_finished() -> None:
    rec = record()
    rec.pop("termination")
    out = verdict(rec)
    assert out.verdict == ERROR and "no ending" in out.reason


def test_a_broken_run_is_the_runs_fault_in_auditkit() -> None:
    """AuditKIT books a run as the target's error, not a scoring error, when stop_reason is 'error'."""
    assert episode_from_record(record(error="RateLimitError")).stop_reason == "error"
    assert episode_from_record(record()).stop_reason == record()["termination"]


def test_a_verdict_saved_with_judges_off_is_not_compared() -> None:
    saved = {"passed": True, "reward": 1.0, "checks": {"DB": 1.0}, "skipped": ["NL_ASSERTION"]}
    assert episode_from_record(record(outcome=saved)).source_verdict is None


def test_the_overall_line_survives_nothing_left_to_score(tmp_path: Any, monkeypatch, capsys) -> None:
    """--only dev on two train files used to crash formatting a missing rate."""
    import json
    import sys

    from _scripts import load_script

    from reflexive import splits

    os_ = load_script("arms/oracle_score")
    files = []
    for name in ("a", "b"):
        f = tmp_path / f"A-retail-{name}.jsonl"
        r = record(domain="retail", task_id=splits.task_ids("retail", "train")[0], error="boom")
        f.write_text(json.dumps(r) + "\n")
        files.append(str(f))
    pytest.importorskip("loguru")
    monkeypatch.setattr(sys, "argv", ["oracle_score.py", *files, "--only", "dev", "--judges", "off"])
    assert os_.main() == 0
    out = capsys.readouterr().out
    assert "overall: success rate n/a (nothing scored)" in out and "no records in the dev split" in out
