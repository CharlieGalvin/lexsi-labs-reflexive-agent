"""Which teacher passes SEL-71's v2 data may add (selection only; no τ²-bench needed)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/data/build_extra_fixes.py"
spec = importlib.util.spec_from_file_location("build_extra_fixes", SCRIPT)
assert spec is not None and spec.loader is not None
bx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bx)

TRAIN = "0"  # airline task 0 is a train task (configs/splits.json)


def fix(task, attempt, passed=True, domain="airline"):
    return {
        "domain": domain,
        "task_id": task,
        "attempt": attempt,
        "sample_model": "glm53f",
        "outcome": {"passed": passed},
        "messages": [],
        "termination": "user_stop",
    }


def base_row(task, group="healable", attempt=1, domain="airline"):
    return {
        "domain": domain,
        "task_id": task,
        "group": group,
        "source": {"golden_id": f"glm53f/{domain}/{task}/{attempt}"},
    }


def test_keeps_new_attempts_on_healable_train_tasks_only(monkeypatch):
    monkeypatch.setattr(bx.golden, "why_not_clean", lambda r: None if r["outcome"]["passed"] else "failed")
    fixes = [fix(TRAIN, 1), fix(TRAIN, 2), fix(TRAIN, 3, passed=False), fix(TRAIN, 2, domain="retail")]
    keep, why = bx.candidates(fixes, [base_row(TRAIN)], {"airline"})
    assert [(r["task_id"], r["attempt"]) for r in keep] == [(TRAIN, 2)]
    assert why["the attempt v1 already uses"] == 1 and why["not a clean pass: failed"] == 1


def test_never_adds_dev_tasks_or_held_out_ones(monkeypatch):
    from reflexive import splits

    monkeypatch.setattr(bx.golden, "why_not_clean", lambda r: None)
    dev = splits.task_ids("airline", "dev")[0]
    keep, why = bx.candidates(
        [fix(dev, 2), fix(TRAIN, 2)], [base_row(TRAIN, group="solved_by_aya")], {"airline"}
    )
    assert keep == []
    assert why["a dev task"] == 1 and why["task not in v1's healable group (held out, or no v1 fix)"] == 1
