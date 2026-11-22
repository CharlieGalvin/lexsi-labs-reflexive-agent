"""Stage 2 training data (scripts/data/build_stage2_data.py, SEL-102). No τ²-bench conversations needed."""

from __future__ import annotations

import pytest
from _scripts import load_script

from reflexive.splits import task_ids

b = load_script("data/build_stage2_data")
TRAIN = sorted(task_ids("retail", "train"))
VAL = {("retail", TRAIN[0])}  # stands in for a validation task


def ex(task, group="mismatch", text="x"):
    return {
        "id": f"{group}/{task}/{text}",
        "format": "reagent.sft/v1",
        "domain": "retail",
        "task_id": task,
        "group": group,
        "messages": [{"role": "system", "content": "s"}, {"role": "assistant", "content": text}],
        "train_on": [1],
    }


def guarded(task, passed=True, blocks=1, termination="agent_stop"):
    return {
        "domain": "retail",
        "task_id": task,
        "outcome": {"passed": passed, "reward": float(passed), "checks": {}},
        "termination": termination,
        "messages": [],
        "monitor": {"actions": {"blocks": [{"action": "blocked"}] * blocks}},
    }


def to_example(r):
    return ex(r["task_id"], group="extra", text=f"guard {r['task_id']}")


BASE = [ex(t, group="healable", text=f"v2 {t}") for t in TRAIN[1:4]]


def test_a_no_go_uses_v2s_data_and_the_mismatch_fixes_only():
    mismatch = [ex(t, text=f"m {t}") for t in TRAIN[1:31]]
    rows, counts = b.build(BASE, mismatch, [guarded(TRAIN[5])], "no-go", to_example)
    assert counts["per_group"] == {"mismatch": 30} and counts["fixes_used"] == 30 and len(rows) == 33
    assert rows[:3] == BASE  # v2's examples first, unchanged


def test_a_go_adds_the_guard_fixes_as_their_own_group():
    mismatch = [ex(t, text=f"m {t}") for t in TRAIN[1:21]]
    guard = [guarded(t, blocks=i % 2) for i, t in enumerate(TRAIN[30:40])]
    rows, counts = b.build(BASE, mismatch, guard, "go", to_example)
    assert counts["per_group"] == {"mismatch": 20, "guard": 10} and counts["fixes_used"] == 30
    assert counts["guard_fixes_with_a_block"] == 5
    g = [r for r in rows if r["group"] == "guard"]
    assert all(r["labels"] == ["stage2-guard"] and r["source"]["kind"] == "guard_fix" for r in g)


def test_under_thirty_fixes_writes_nothing():
    with pytest.raises(b.TooFew, match="29 fixes"):
        b.build(BASE, [ex(t, text=f"m {t}") for t in TRAIN[1:30]], [], "no-go", to_example)


def test_repeats_are_left_out_and_dont_count():
    mismatch = [ex(t, text=f"m {t}") for t in TRAIN[1:31]] + [ex(TRAIN[1], text=f"m {TRAIN[1]}")]
    _, counts = b.build(BASE, mismatch, [], "no-go", to_example)
    assert counts["fixes_used"] == 30 and counts["repeats_left_out"] == {"mismatch": 1}


def test_only_clean_passing_guarded_train_conversations_become_guard_fixes():
    recs = [
        guarded(TRAIN[0]),
        guarded(TRAIN[5], passed=False),
        guarded(TRAIN[6], termination="max_steps"),
        guarded(TRAIN[7]),
    ]
    keep, why = b.select_guard(recs, VAL)
    assert [r["task_id"] for r in keep] == [TRAIN[7]]
    assert why["a validation task"] == 1 and why["failed a check"] == 1 and sum(why.values()) == 3


def test_mismatch_fixes_must_be_train_tasks_in_their_group():
    assert b.check_mismatch([ex(TRAIN[2])], VAL) == []
    bad = b.check_mismatch(
        [ex(TRAIN[0]), ex(TRAIN[2], group="onpolicy"), {**ex(TRAIN[3]), "train_on": [5]}], VAL
    )
    assert any("a validation task" in x for x in bad) and any("group 'onpolicy'" in x for x in bad)
    assert any("train_on" in x for x in bad)
