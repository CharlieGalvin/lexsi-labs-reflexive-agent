"""reflexive.oracle: τ²-bench's scoring as an AuditKIT outcome oracle (SEL-46).

The first tests are offline. The rest replay each task's correct actions through τ²-bench's
evaluator (no model, no API calls; judges off) and are skipped without τ²-bench.
"""

from __future__ import annotations

from typing import Any

import pytest
from auditkit.agent_eval.outcome import ERROR, FAILURE, SUCCESS, UNKNOWN, resolve_outcome
from test_golden import record

from reflexive.oracle import Tau2Oracle, case_for, episode_from_record, record_of

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
    assert on["type"] == "tau2_score" and on["fn"] == "reflexive.outcome.score"
    assert "tau2-bench 2.0.0" in on["version"] and on["version"] != off["version"]
    assert "judges off" in off["version"]


def test_the_episode_shows_what_the_agent_saw_and_keeps_the_record_for_scoring() -> None:
    rec = record(outcome={"passed": True, "reward": 1.0, "checks": {}, "skipped": []})
    ep = episode_from_record(rec)
    calls = [e for e in ep.events if e.type == "tool_call"]
    assert calls and all("check_status_bar" not in str(e.payload) for e in calls)  # the customer's own action
    assert record_of(ep)["messages"] == rec["messages"] and ep.stop_reason == rec["termination"]
    assert "tau2_record" not in ep.metadata and "tau2_record" not in ep.to_dict()
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


# --- Review of #43 (Chiara, 6 Nov) -------------------------------------------------------------


def test_a_pass_with_judged_checks_skipped_is_unverified_even_when_no_check_ran() -> None:
    """A judged-only task with judges off: τ²-bench runs nothing and returns 1.0."""
    from reflexive.outcome import Outcome

    nothing_ran = Outcome(passed=True, reward=1.0, checks={}, judged=False, skipped=["NL_ASSERTION"])
    assert not nothing_ran.verified
    failed = Outcome(passed=False, reward=0.0, checks={"DB": 0.0}, judged=False, skipped=["NL_ASSERTION"])
    assert failed.verified  # a check that ran failed: the skipped ones can't save it
    assert Outcome(passed=True, reward=1.0, checks={"DB": 1.0}, judged=True).verified


def test_the_oracle_calls_a_judged_only_pass_unknown(monkeypatch) -> None:
    import reflexive.outcome as outcome

    fake = outcome.Outcome(passed=True, reward=1.0, checks={}, judged=False, skipped=["COMMUNICATE"])
    monkeypatch.setattr(outcome, "score", lambda *a, **k: fake)
    out = verdict(record())
    assert out.verdict == UNKNOWN and "no check ran" in out.reason


def test_a_failure_saved_with_judges_off_is_still_compared() -> None:
    saved = {"passed": False, "reward": 0.0, "checks": {"DB": 0.0}, "skipped": ["NL_ASSERTION"]}
    assert episode_from_record(record(outcome=saved)).source_verdict == FAILURE


def test_a_crash_with_no_ending_keeps_its_error_text() -> None:
    rec = record(error="RateLimitError: 429")
    rec.pop("termination")
    out = verdict(rec)
    assert out.verdict == ERROR and "RateLimitError" in out.reason and out.evidence["error"]


def test_the_episode_carries_only_what_the_oracle_scores_from() -> None:
    ep = episode_from_record(record())
    assert set(record_of(ep)) == {"domain", "task_id", "messages", "termination", "error"}


# --- just oracle-score (review of #43) ---------------------------------------------------------


def _script() -> Any:
    from _scripts import load_script

    pytest.importorskip("loguru")
    return load_script("arms/oracle_score")


def _main(monkeypatch, *argv: str) -> int:
    import sys

    monkeypatch.setattr(sys, "argv", ["oracle_score.py", *argv])
    return _script().main()


def _write(path: Any, *recs: Record, tail: str = "") -> str:
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in recs) + tail)
    return str(path)


def test_a_malformed_record_is_an_error_verdict_not_a_traceback() -> None:
    os_ = _script()
    bad = record()
    bad.pop("messages")
    [(_, out, saved)] = os_.score_all([bad], Tau2Oracle(judges=False))
    assert out.verdict == ERROR and "can't be read" in out.reason and saved is None


def test_scoring_in_parallel_keeps_the_records_order() -> None:
    os_ = _script()
    recs = [record(task_id=str(i), error="boom") for i in range(20)]
    got = os_.score_all(recs, Tau2Oracle(judges=False), workers=8)
    assert [r["task_id"] for r, _, _ in got] == [str(i) for i in range(20)]


def test_a_half_written_last_line_is_skipped(tmp_path: Any, monkeypatch, capsys) -> None:
    f = _write(tmp_path / "A-telecom-train.jsonl", record(error="boom"), tail='{"domain": "tel')
    assert _main(monkeypatch, f, "--judges", "off") == 0
    assert "1 error" in capsys.readouterr().out


def test_judges_on_without_the_key_is_refused_and_auto_falls_back(tmp_path: Any, monkeypatch, capsys) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    f = _write(tmp_path / "A-telecom-train.jsonl", record(error="boom"))
    assert _main(monkeypatch, f, "--judges", "on") == 1 and "needs OPENAI_API_KEY" in capsys.readouterr().out
    assert _main(monkeypatch, f) == 0 and "judges off (auto: no OPENAI_API_KEY)" in capsys.readouterr().out


def test_the_same_file_twice_is_refused_and_repeats_count_once(tmp_path: Any, monkeypatch, capsys) -> None:
    f = _write(tmp_path / "a.jsonl", record(error="boom"))
    assert _main(monkeypatch, f, f, "--judges", "off") == 1 and "given twice" in capsys.readouterr().out
    g = _write(tmp_path / "b.jsonl", record(error="boom"), record(error="boom", sample=1))
    assert _main(monkeypatch, f, g, "--judges", "off") == 0
    out = capsys.readouterr().out
    assert "1 conversations already in an earlier file (e.g. ? telecom:t1 #0)" in out
    assert "overall: success rate n/a (nothing scored), 0 unknown, 2 error" in out


def test_only_leaves_out_a_task_in_no_split_with_a_warning(tmp_path: Any, monkeypatch, capsys) -> None:
    f = _write(tmp_path / "a.jsonl", record(domain="retail", task_id="9999"))
    assert _main(monkeypatch, f, "--only", "dev", "--judges", "off") == 0
    assert "1 records in no frozen split" in capsys.readouterr().out


def test_disagreements_name_the_sample() -> None:
    from types import SimpleNamespace as NS

    os_ = _script()
    r = {"arm": "A", "domain": "retail", "task_id": "12", "sample": 2}
    s = os_.summary([(r, NS(verdict=FAILURE, reason="DB"), SUCCESS)])
    assert s["disagreements_with_saved"][0]["task"] == "A retail:12 #2"
