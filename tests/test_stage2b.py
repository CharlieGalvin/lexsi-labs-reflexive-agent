"""Stage 2b (STAGE2.md): the decision rule (scripts/arms/stage2b_compare.py) and the training-data
audit's reading of Cohere-format records (scripts/data/audit_training_data.py). No API calls."""

from __future__ import annotations

import json

import pytest
from _scripts import load_script

from reflexive.splits import task_ids
from reflexive.teacher import to_prompted

cmp = load_script("arms/stage2b_compare")
aud = load_script("data/audit_training_data")


def tasks(d, n):
    return [(d, str(t)) for t in sorted(task_ids(d, "train"))[:n]]


def table(passes):
    """{task: [bool]*4} from {task: number of passes of 4}."""
    return {t: [i < k for i in range(4)] for t, k in passes.items()}


def test_a_clear_gain_meets_the_rule():
    ts = tasks("retail", 13) + tasks("airline", 5) + tasks("telecom", 12)
    base = table({t: 2 for t in ts})
    cand = table({t: (3 if i % 3 == 0 else 2) for i, t in enumerate(ts)})  # +0.25 on 10 tasks: +2.5
    res = cmp.decide(base, cand, 4, 2000)
    assert res["gain"] == 2.5 and res["ci95"][0] > 0 and res["meets_rule"]


def test_a_domain_drop_or_a_small_gain_fails_and_a_missing_sample_blocks_the_verdict():
    ts = tasks("retail", 13) + tasks("telecom", 12)
    base = table({t: 2 for t in ts})
    cand = table({t: (4 if t[0] == "retail" else 1) for t in ts})  # retail +26/4, telecom −12/4
    res = cmp.decide(base, cand, 4, 2000)
    assert res["domains"]["telecom"]["change"] == -3.0 and not res["meets_rule"]
    small = cmp.decide(base, table({t: (3 if t == ts[0] else 2) for t in ts}), 4, 2000)
    assert small["gain"] == 0.25 and not small["checks"]["gain at least +2.0"]
    cand[ts[0]] = cand[ts[0]][:3]
    gap = cmp.decide(base, cand, 4, 2000)
    assert gap["incomplete"] and not gap["meets_rule"] and gap["tasks"] == len(ts) - 1


def call(name, i, **args):
    return {"id": f"x{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


OPENAI = [
    {"role": "system", "content": "POLICY"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Cancel order #W1 please."},
    {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", 0, order_id="#W1")]},
    {"role": "tool", "tool_call_id": "x0", "content": '{"status": "pending"}'},
    {"role": "assistant", "content": "It's pending. Cancel it? (yes/no)"},
    {"role": "user", "content": "Yes."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("cancel_pending_order", 1, order_id="#W1", reason="no longer needed")],
    },
    {"role": "tool", "tool_call_id": "x1", "content": '{"status": "cancelled"}'},
    {"role": "assistant", "content": "Done."},
]


def test_a_cohere_training_record_reads_back_as_the_conversation_the_checker_needs():
    back = aud.to_openai(to_prompted(OPENAI, [], "cohere"))
    assert [m["role"] for m in back] == [m["role"] for m in OPENAI]
    calls = [c["function"] for m in back for c in m.get("tool_calls") or []]
    assert [c["name"] for c in calls] == ["get_order_details", "cancel_pending_order"]
    assert json.loads(calls[1]["arguments"]) == {"order_id": "#W1", "reason": "no longer needed"}
    tools = [m for m in back if m["role"] == "tool"]
    assert tools[1]["tool_call_id"] == back[7]["tool_calls"][0]["id"] and "cancelled" in tools[1]["content"]
    assert back[0]["content"].startswith("POLICY")


def test_the_audit_counts_conversations_with_a_live_finding_and_keeps_errors_apart():
    t = sorted(task_ids("retail", "train"))[0]
    exs = [
        {
            "id": f"g/retail/{t}/{i}",
            "group": "healable",
            "domain": "retail",
            "task_id": t,
            "messages": to_prompted(OPENAI, [], "cohere"),
        }
        for i in range(3)
    ]
    answers = iter(
        [
            [{"live": True, "rule": "no_yes", "turn": 7, "tool": "cancel_pending_order", "reason": "no yes"}],
            [{"live": False, "rule": "no_yes_judge_error", "reason": "timeout"}],
            [],
        ]
    )
    counts, flagged = aud.audit(exs, lambda rec: next(answers))
    assert counts["all"] == {
        "conversations": 3,
        "flagged": 1,
        "no_yes": 1,
        "mismatch": 0,
        "checker_errors": 1,
    }
    assert [f["id"] for f in flagged] == [exs[0]["id"]]


def test_the_audit_refuses_a_task_outside_train():
    t = sorted(task_ids("retail", "test"))[0]
    with pytest.raises(SystemExit):
        aud.audit([{"domain": "retail", "task_id": t, "messages": []}], lambda rec: [])
