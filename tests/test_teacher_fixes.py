"""scripts/teacher_fixes.py: the fix run's task list, attempts and report (SEL-58 phase 2)."""

from __future__ import annotations

import importlib.util
import pathlib

import pytest


def load():
    spec = importlib.util.spec_from_file_location(
        "teacher_fixes", pathlib.Path(__file__).parents[1] / "scripts" / "teacher_fixes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(domain, task, attempt, passed, checks=None, broken=False, judged=True):
    return {"domain": domain, "task_id": task, "attempt": attempt, "termination": "user_stop",
            "outcome": {"passed": passed, "judged": judged, "checks": checks or {"DB": 1.0}},
            "diagnostics": {"repeated_failures": 0, "unknown_tools": 0, "unseen_arguments": []},
            "agent_usage": {"est_cost": 0.01}, "user_cost": 0.02,
            "error": "boom" if broken else None}


def test_retries_are_warmer_than_the_first_attempt():
    fx = load()
    assert fx.TEMPERATURES[0] == 0.0 and all(t > 0 for t in fx.TEMPERATURES[1:])


def test_db_only_ignores_judged_checks():
    fx = load()
    assert fx.db_only_pass(rec("retail", "1", 1, False, {"DB": 1.0, "NL_ASSERTION": 0.0}))
    assert not fx.db_only_pass(rec("retail", "1", 1, False, {"DB": 0.0}))


def test_report_counts_pass_at_3_and_pass_all_3():
    fx = load()
    recs = [rec("retail", "1", 1, True), rec("retail", "1", 2, True), rec("retail", "1", 3, True),
            rec("retail", "2", 1, False), rec("retail", "2", 2, True), rec("retail", "2", 3, False),
            rec("telecom", "t", 1, False), rec("telecom", "t", 2, False),
            rec("telecom", "t", 3, False, broken=True)]
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
    assert not set(tasks) & set(fx.ts.sample_tasks())
    assert {d: sum(x == d for x, _ in tasks) for d in fx.DOMAINS} == {"retail": 54, "airline": 20,
                                                                       "telecom": 74}
