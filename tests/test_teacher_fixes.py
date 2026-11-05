"""scripts/teacher/fixes.py: the fix run's task list, attempts and report (SEL-58 phase 2)."""

from __future__ import annotations

import pytest
from _scripts import load_script


def load():
    return load_script("teacher/fixes")


def rec(domain, task, attempt, passed, checks=None, broken=False, judged=True):
    return {
        "domain": domain,
        "task_id": task,
        "attempt": attempt,
        "termination": "user_stop",
        "outcome": {"passed": passed, "judged": judged, "checks": checks or {"DB": 1.0}},
        "diagnostics": {"repeated_failures": 0, "unknown_tools": 0, "unseen_arguments": []},
        "agent_usage": {"est_cost": 0.01},
        "user_cost": 0.02,
        "error": "boom" if broken else None,
    }


def test_retries_are_warmer_than_the_first_attempt():
    fx = load()
    assert fx.TEMPERATURES[0] == 0.0 and all(t > 0 for t in fx.TEMPERATURES[1:])


def test_db_only_ignores_judged_checks():
    fx = load()
    assert fx.db_only_pass(rec("retail", "1", 1, False, {"DB": 1.0, "NL_ASSERTION": 0.0}))
    assert not fx.db_only_pass(rec("retail", "1", 1, False, {"DB": 0.0}))


def test_report_counts_pass_at_3_and_pass_all_3():
    fx = load()
    recs = [
        rec("retail", "1", 1, True),
        rec("retail", "1", 2, True),
        rec("retail", "1", 3, True),
        rec("retail", "2", 1, False),
        rec("retail", "2", 2, True),
        rec("retail", "2", 3, False),
        rec("telecom", "t", 1, False),
        rec("telecom", "t", 2, False),
        rec("telecom", "t", 3, False, broken=True),
    ]
    r = fx.report(recs)
    assert r["retail"]["pass1"] == (1, 2)
    assert r["retail"]["pass_at_3"] == (2, 2) and r["retail"]["pass_all_3"] == (1, 2)
    assert r["retail"]["clean_fixes"] == 4 and r["all"]["broken"] == 1
    assert r["all"]["tasks"] == 3 and round(r["all"]["cost"], 2) == 0.27


def test_remaining_train_tasks_are_the_148_not_used_in_phase_1():
    pytest.importorskip("tau2")
    from tau2.utils.utils import DATA_DIR

    if not (DATA_DIR / "tau2" / "domains").exists():
        pytest.skip("τ²-bench data missing")
    fx = load()
    tasks = fx.remaining_train_tasks()
    assert len(tasks) == 148 and len(set(tasks)) == 148
    assert not set(tasks) & set(fx.runner.sample_tasks())
    assert {d: sum(x == d for x, _ in tasks) for d in fx.DOMAINS} == {
        "retail": 54,
        "airline": 20,
        "telecom": 74,
    }


def test_preflight_reports_every_failing_model(monkeypatch):
    import sys
    import types

    def completion(model, **kw):
        if model == "bad":
            raise RuntimeError("You have no credits remaining.")

    monkeypatch.setitem(
        sys.modules, "litellm", types.SimpleNamespace(completion=completion, suppress_debug_info=False)
    )
    fx = load()
    problems = fx.preflight(["good", "bad", "bad"])
    assert len(problems) == 1 and "add OpenAI credits" in problems[0]


def test_needs_judging_picks_failed_or_skipped_judges():
    fx = load()
    ok = rec("retail", "1", 1, True)
    ok["outcome"]["skipped"] = []
    skipped = rec("retail", "2", 1, True)
    skipped["outcome"]["skipped"] = ["NL_ASSERTION"]
    failed = {**rec("retail", "3", 1, True), "judge_error": "RateLimitError"}
    broken = rec("retail", "4", 1, False, broken=True)
    assert [fx.needs_judging(r) for r in (ok, skipped, failed, broken)] == [False, True, True, False]


def test_pass_at_3_counts_tasks_that_stopped_at_their_first_pass():
    fx = load()
    recs = [
        {**rec("retail", "1", 1, True), "attempt_mode": "until_pass"},
        *({**rec("retail", "2", i, i == 2), "attempt_mode": "until_pass"} for i in (1, 2)),
        *({**rec("retail", "3", i, False), "attempt_mode": "until_pass"} for i in (1, 2, 3)),
    ]
    r = fx.report(recs)["retail"]
    assert r["pass1"] == (1, 3) and r["pass_at_3"] == (2, 3)
    assert r["pass_all_3"] == (0, 0)  # not measured when retries stop at a pass


def test_view_check_spots_customer_actions_shown_to_the_agent():
    sr = load_script("student/review")
    call = {"id": "u1", "function": {"name": "check_status_bar"}}
    full = [
        {"role": "user", "content": None, "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "u1", "content": "No signal"},
        {"role": "user", "content": "It says no signal."},
    ]
    hidden = {"domain": "telecom", "messages": full, "agent_view": [full[2]]}
    shown = {"domain": "telecom", "messages": full, "agent_view": full}
    assert sr.view_check([hidden]) == (1, 1, 0) and sr.view_check([shown]) == (1, 1, 1)
