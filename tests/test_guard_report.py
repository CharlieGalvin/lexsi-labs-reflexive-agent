"""Decision 1's guard report (scripts/arms/guard_report.py): what counts as a wrongly blocked good
change, on hand-made records (no τ²-bench, no API calls)."""

from __future__ import annotations

from _scripts import load_script

g = load_script("arms/guard_report")
EXPECTED = {
    ("retail", "1"): [("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}, None)]
}


def expected(d, t):
    return EXPECTED.get((d, t), [])


def rec(task="1", passed=True, blocks=(), off=False, domain="retail", sample=0):
    blocks = list(blocks)
    return {
        "domain": domain,
        "task_id": task,
        "sample": sample,
        "termination": "user_stop",
        "messages": [{"role": "assistant", "content": "Hi"}],
        "outcome": {"passed": passed, "reward": float(passed)},
        "monitor": {
            "actions": {
                "counts": {
                    a: sum(b["action"] == a for b in blocks) for a in ("blocked", "overridden", "held")
                },
                "blocks": blocks,
                "reasks": [{} for b in blocks if b["action"] == "blocked"],
                "switched_off_by_error": off,
            }
        },
    }


def block(args, action="blocked", rule="mismatch", also=()):
    return {
        "tool": "cancel_pending_order",
        "arguments": args,
        "action": action,
        "rule": rule,
        "also": list(also),
    }


GOOD = {"order_id": "#W1", "reason": "no longer needed"}
BAD = {"order_id": "#W2", "reason": "no longer needed"}


def test_only_a_blocked_expected_change_in_a_passing_conversation_counts():
    recs = [
        rec(blocks=[block(GOOD)], sample=0),  # wrong block: τ²-bench expects it, and it passed
        rec(blocks=[block(BAD)], sample=1),  # a block of a change τ²-bench doesn't expect: fine
        rec(task="2", blocks=[block(GOOD)]),  # another task: not expected there
        rec(passed=False, blocks=[block(GOOD)], sample=2),  # a failure: not in the denominator
        rec(blocks=[block(GOOD, action="overridden")], sample=3),  # ran anyway: not a block
        rec(blocks=[block('{"order_id": "#W1", "reason": "no longer needed"}')], sample=4),  # JSON text
    ]
    rep = g.report(recs, expected)
    assert rep["all"]["passed"] == 5 and rep["all"]["passed_with_wrong_block"] == 2
    assert rep["all"]["wrong_block_rate"] == 0.4 and rep["all"]["meets_bar"] is False
    assert rep["all"]["blocked"] == 5 and rep["all"]["overridden"] == 1 and rep["all"]["reasks"] == 5


def test_the_bar_is_at_most_one_in_ten_and_costs_are_counted():
    recs = [rec(sample=s) for s in range(9)] + [rec(blocks=[block(GOOD)], sample=9)]
    recs.append(rec(task="3", off=True, blocks=[block(BAD, also=[{"rule": "no_yes_judge_error"}])]))
    rep = g.report(recs, expected)
    assert rep["all"]["passed_with_wrong_block"] == 1 and rep["all"]["passed"] == 11
    assert rep["all"]["meets_bar"] is True
    assert rep["retail"]["guard_switched_off"] == 1 and rep["retail"]["judge_errors"] == 1


def test_a_crashed_conversation_is_left_out():
    crashed = rec(blocks=[block(GOOD)])
    crashed["termination"] = "error"
    crashed["error"] = "rate limit"
    assert g.report([crashed, rec(sample=1)], expected)["all"]["conversations"] == 1
