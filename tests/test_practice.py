"""The self-practice round (scripts/heal/practice.py): which practice conversations are kept, and
that they're formatted as v2's examples. Hand-made records; no τ²-bench run, no API calls."""

from __future__ import annotations

import json

import pytest
from _scripts import load_script

from reflexive.splits import task_ids

pr = load_script("heal/practice")
TRAIN = sorted(str(t) for t in task_ids("retail", "train"))
VAL = {("retail", TRAIN[0])}


def call(name, i, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


MESSAGES = [
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What's the status of #W1?"},
    {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", 0, order_id="#W1")]},
    {"role": "tool", "tool_call_id": "c0", "content": '{"status": "pending"}'},
    {"role": "assistant", "content": "It's pending."},
    {"role": "user", "content": "Thanks! ###STOP###"},
]


def rec(task, sample=0, passed=True, **kw):
    return {
        "domain": "retail",
        "task_id": task,
        "sample": sample,
        "termination": "user_stop",
        "outcome": {"passed": passed, "reward": float(passed), "checks": {}},
        "agent_view": [{"role": "system", "content": "SYSTEM"}, *MESSAGES],
        "messages": MESSAGES,
        **kw,
    }


def test_the_pool_is_the_train_tasks_outside_validation():
    ids = pr.pool("retail", {TRAIN[0]})
    assert TRAIN[0] not in ids and len(ids) == len(TRAIN) - 1


def test_only_clean_unflagged_passes_are_kept_at_most_two_per_task():
    t = TRAIN[1]
    recs = [rec(t, 0), rec(t, 1), rec(t, 2), rec(TRAIN[2], 0, passed=False), rec(TRAIN[3], 0)]
    flagged = {(TRAIN[3], 0)}
    keep, why, passed = pr.select(
        recs, VAL, 2, lambda r: [{}] if (r["task_id"], r["sample"]) in flagged else []
    )
    assert [(r["task_id"], r["sample"]) for r in keep] == [(t, 0), (t, 1)]
    assert why["failed"] == 1 and why['"no yes" flagged a change'] == 1 and why["over 2 per task"] == 1
    assert passed["retail"] == {t}


def test_a_validation_task_in_the_practice_files_stops_the_build():
    with pytest.raises(SystemExit):
        pr.select([rec(TRAIN[0])], VAL, 2, None)


def test_a_kept_conversation_is_formatted_as_v2s_examples_are():
    ex = pr.to_example(rec(TRAIN[1], 1), 0, "S2PR")
    assert ex["group"] == "practice" and ex["id"] == f"practice/S2PR/retail/{TRAIN[1]}/s1"
    roles = [m["role"] for m in ex["messages"]]
    assert roles[0] == "system" and ex["train_on"] == [i for i, r in enumerate(roles) if r == "assistant"]
    assert any(m["content"].startswith("Action: ```json") for m in ex["messages"])
    assert ex["messages"][0]["content"].startswith("SYSTEM")
