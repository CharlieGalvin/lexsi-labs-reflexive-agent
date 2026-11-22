"""Meta-healing v2 (reflexive.meta_heal_v2). Every model is a fake: no API calls."""

from __future__ import annotations

import json

import pytest

from reflexive import meta_heal as mh
from reflexive import meta_heal_v2 as v2

KEY = "ANSWER-KEY-SENTINEL"


def call(name="cancel_pending_order", **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def rec(task_id="1", passed=False, sample=0, domain="retail"):
    long_result = '{"items": [' + ", ".join(f'{{"item_id": "{i:010d}"}}' for i in range(90)) + "]}"
    return {
        "domain": domain,
        "task_id": task_id,
        "sample": sample,
        "outcome": {"passed": passed, "checks": {"db": float(passed)}},
        "agent_view": [
            {"role": "system", "content": "POLICY TEXT"},
            {"role": "user", "content": "Cancel order #W1."},
            {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", order_id="#W1")]},
            {"role": "tool", "content": long_result},
            {"role": "assistant", "content": "Cancelling.", "tool_calls": [call(order_id="#W1", reason="x")]},
        ],
    }


def expected(r):
    return [f'cancel_pending_order({{"order_id": "{KEY}"}})']


class Fake:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def __call__(self, messages):
        self.seen.append(messages)
        return self.answer(messages) if callable(self.answer) else self.answer


RULE = {"name": "r1", "what": "w", "applies_to": "call", "tools": ["cancel_pending_order"], "question": "q?"}


def test_the_proposer_sees_the_answer_key_and_full_tool_results_and_no_error_kind_is_named():
    msgs = v2.proposer_request([rec()], [rec("2", passed=True)], expected)
    user = msgs[1]["content"]
    assert KEY in user and "EXPECTED ACTIONS" in user and "0000000089" in user  # past v1's 600 characters
    assert user.count("EXPECTED ACTIONS") == 1  # only failures carry the key
    assert (
        "mismatch" not in msgs[0]["content"].lower() and "match what the customer" not in msgs[0]["content"]
    )


def test_rules_are_checked_by_full_gpt41_and_never_see_the_answer_key():
    (c,) = v2.parse({"rules": [RULE]})
    assert c.model == "gpt-4.1-2025-04-14" and c.tool_chars == 6000 and c.tools == ("cancel_pending_order",)
    chk = Fake({"violated": False})
    v2.findings_by_record([c], [rec()], chk, workers=1)
    assert chk.seen and all(KEY not in json.dumps(m) and "EXPECTED" not in json.dumps(m) for m in chk.seen)


def test_learning_rounds_feed_back_what_each_rule_caught_and_wrongly_flagged():
    recs = [rec("1"), rec("2"), rec("3", passed=True), rec("4", passed=True)]
    rounds = iter(
        [{"rules": [RULE]}, {"rules": [{**RULE, "name": "r2"}]}, {"rules": [{**RULE, "name": "r3"}]}]
    )
    proposer = Fake(lambda m: next(rounds))
    chk = Fake({"violated": True, "evidence": "Cancel order #W1.", "reason": "because"})
    out = v2.learn(
        recs,
        proposer,
        chk,
        rounds=3,
        n_passes_scored=2,
        expected=expected,
        explained=lambda r: r["task_id"] == "2",
        log=lambda s: None,
    )
    assert out["unexplained_failures"] == ["retail:1#0"]
    assert [r["round"] for r in out["rounds"]] == [1, 2, 3] and out["final"][0]["name"] == "r3"
    assert out["rounds"][0]["counts"]["r1"] == {"failures": [1, 1], "passes": [2, 2]}
    second = proposer.seen[1]
    assert second[-2]["role"] == "assistant" and "r1: caught 1/1 failures" in second[-1]["content"]
    assert "wrongly flagged 2/2 passes" in second[-1]["content"] and "because" in second[-1]["content"]
    assert len(proposer.seen) == 3


def test_the_held_out_bar_counts_added_catches_passes_at_worst_and_rediscovery():
    (c,) = v2.parse({"rules": [RULE]})
    recs = [rec(str(i)) for i in range(10)] + [rec(f"p{i}", passed=True) for i in range(10)]
    frozen = {
        mh.example_id(r): {"tal": False, "no_yes": False, "mismatch": r["task_id"] in {"0", "1"}}
        for r in recs
    }

    def answer(m):
        text = m[1]["content"]
        return (
            {"violated": True, "evidence": "Cancel order #W1.", "reason": "r"}
            if "flag-me" in text
            else {"violated": False}
        )

    for r in recs[2:5] + recs[10:11]:  # 3 added catches, 1 pass
        r["agent_view"][1]["content"] = "Cancel order #W1. flag-me"
    (s,) = v2.held_out([c], recs, frozen, Fake(answer), workers=1)
    assert s.added == [3, 8] and s.failures == [3, 10] and s.passes == [1, 10] and s.rediscovers == [0, 2]
    assert s.meets_bar and v2.shortlist([s]) == ["r1"]
    s.passes_errored = 1
    assert not s.meets_bar  # 2/10 passes at worst


def test_the_script_allows_at_most_three_rounds():
    from _scripts import load_script

    with pytest.raises(SystemExit, match="at most 3"):
        load_script("heal/meta_heal_v2").main(["learn", "--rounds", "4"])


def test_the_layers_table_is_cumulative_and_counts_failures_and_passes():
    ids = {"f1": False, "f2": False, "f3": False, "f4": False, "p1": True, "p2": True}
    frozen = {
        "f1": {"tal": True, "no_yes": False, "mismatch": False},
        "f2": {"tal": False, "no_yes": True, "mismatch": True},
        "f3": {"tal": False, "no_yes": False, "mismatch": True},
        "f4": {"tal": False, "no_yes": False, "mismatch": False},
        "p1": {"tal": False, "no_yes": False, "mismatch": True},
        "p2": {"tal": False, "no_yes": False, "mismatch": False},
    }
    scores = [
        {"name": "r1", "flagged": [{"id": "f4"}, {"id": "f3"}]},
        {"name": "r2", "flagged": [{"id": "p2"}]},
    ]
    rows = v2.layers(ids, frozen, scores, ["r1"])
    assert [r["failures"] for r in rows] == [[1, 4], [2, 4], [3, 4], [4, 4]]
    assert [r["passes"] for r in rows] == [[0, 2], [0, 2], [1, 2], [1, 2]]  # r2 isn't accepted
    assert len(v2.layers(ids, frozen, scores, [])) == 3  # nothing accepted: no fourth layer
