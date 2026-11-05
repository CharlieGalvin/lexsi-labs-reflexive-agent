"""scripts/teacher/bakeoff.py: the bake-off's summary and teacher-vs-student overlap."""

from __future__ import annotations

import pytest
from _scripts import load_script

from reflexive import runner
from reflexive.episode import official_split


def load():
    return load_script("teacher/bakeoff")


def test_summary_counts_conversations_with_each_problem():
    ts = load()

    def rec(domain, passed, unseen=(), unknown=0, retried=0, tools=3, ended="user_stop"):
        return {"domain": domain, "outcome": {"passed": passed}, "error": None, "user_cost": 0.05,
                "termination": ended,
                "wall_s": 60.0, "agent_usage": {"est_cost": 0.05},
                "diagnostics": {"unseen_arguments": list(unseen), "unknown_tools": unknown,
                                "repeated_failures": retried, "tools_ran": tools}}

    s = ts.summarise("aya", [rec("retail", True), rec("retail", False, unseen=["1012345678"], retried=2),
                             rec("airline", False, unknown=4, tools=0),
                             rec("airline", False, tools=0, ended="unexpected_error")])
    assert (s["passed"], s["n"]) == (1, 4) and s["by_domain"] == {"retail": (1, 2), "airline": (0, 2)}
    assert (s["invented_ids"], s["unknown_tools"], s["repeated_failures"], s["no_tools"]) == (1, 1, 1, 2)
    assert s["errors"] == 1  # the broken run is reported, not silently counted as a model failure
    assert s["cost"] == pytest.approx(0.40)


def test_phase1_sample_is_fixed_train_only_and_needs_a_state_change():
    pytest.importorskip("tau2")

    from reflexive.outcome import scorer

    tasks = runner.sample_tasks()
    assert tasks == runner.sample_tasks()  # same every run
    assert sum(d == "retail" for d, _ in tasks) == 20 and sum(d == "airline" for d, _ in tasks) == 10
    for d, t in tasks:
        assert t in official_split(d, "train")
        assert scorer(d).write_action_indices(t)


def test_overlap_lists_tasks_only_the_teacher_solves(capsys):
    ts = load()
    ts.print_overlap({"qwen": {("retail", "4"), ("airline", "7")},
                      "tiny": {("airline", "7"), ("retail", "9")}})
    out = capsys.readouterr().out
    assert "qwen → tiny   1   (1 the other way, 1 both)  retail:4" in out
