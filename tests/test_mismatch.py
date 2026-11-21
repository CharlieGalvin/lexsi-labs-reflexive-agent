"""The hand-written "mismatch" rule (reflexive.mismatch) and its injected-error test. No API calls."""

from __future__ import annotations

import json
import random

from _scripts import load_script

from reflexive import meta_heal as mh
from reflexive.actions import RULES
from reflexive.intent import ALL_WRITE_TOOLS
from reflexive.mismatch import RULE, TOOLS, mismatch


def call(name, **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


LONG = '{"order_id": "#W1", "items": [' + ", ".join(f'{{"item_id": "{i:010d}"}}' for i in range(150)) + "]}"
VIEW = [
    {"role": "system", "content": "POLICY"},
    {"role": "user", "content": "Swap the red one for the blue one in order #W1."},
    {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", order_id="#W1")]},
    {"role": "tool", "content": LONG},
]


class Fake:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def __call__(self, messages):
        self.seen.append(messages)
        return self.answer


def test_it_is_registered_after_no_yes_and_is_live_with_its_own_message():
    assert [r.__name__ for r in RULES][:2] == ["no_yes", "mismatch"]
    assert RULE.live and RULE.block_message and RULE.applies_to == "call"
    assert RULE.model == "gpt-4.1-2025-04-14"  # full GPT-4.1, pinned
    assert mh.default_rule_llm() is None  # conftest: judges off


def test_it_checks_only_the_listed_tools_and_never_fires_with_judges_off():
    yes = Fake({"violated": True, "evidence": "red one", "reason": "r"})
    rule = mh.as_rule(RULE, llm=yes, kind="INTENT", prefix="")
    assert rule(VIEW, call("modify_pending_order_address", order_id="#W1"), "p") is None and not yes.seen
    f = rule(
        VIEW, call("modify_pending_order_items", order_id="#W1", item_ids=["1"], new_item_ids=["2"]), "p"
    )
    assert f is not None and (f.type, f.rule, f.live) == ("INTENT", "mismatch", True)
    assert f.message.startswith("Note from the system, not the customer: this change may not match")
    assert mismatch(VIEW, call("cancel_pending_order", order_id="#W1"), "p") is None  # conftest: judges off
    assert set(TOOLS) <= ALL_WRITE_TOOLS


def test_the_checker_sees_long_tool_results_in_full():
    f = Fake({"violated": False})
    mh.as_rule(RULE, llm=f)(VIEW, call("cancel_pending_order", order_id="#W1"), "p")
    assert '"item_id": "0000000149"' in f.seen[0][1]["content"]  # past meta-healing's 1,200 characters


def test_corruptions_change_the_call_in_one_known_way():
    s = load_script("labels/rule_rates")
    rng = random.Random(0)
    seen = '"item_id": "1111111111" "item_id": "2222222222" "item_id": "3333333333" #W1234567 #W7654321'
    c, kind = s.corrupt(
        call(
            "exchange_delivered_order_items",
            order_id="#W1234567",
            item_ids=["1111111111"],
            new_item_ids=["2222222222"],
        ),
        seen,
        rng,
    )
    assert json.loads(c["function"]["arguments"])["new_item_ids"] == ["3333333333"] and "swapped" in kind
    c, kind = s.corrupt(call("cancel_pending_order", order_id="#W1234567", reason="x"), seen, rng)
    assert json.loads(c["function"]["arguments"])["order_id"] == "#W7654321"
    fl = [{"flight_number": "HAT1", "date": "2024-05-19"}, {"flight_number": "HAT2", "date": "2024-05-23"}]
    c, kind = s.corrupt(call("update_reservation_flights", reservation_id="ABC", flights=fl), "", rng)
    assert json.loads(c["function"]["arguments"])["flights"] == fl[:1] and kind == "a flight dropped"
    c, _ = s.corrupt(call("update_reservation_flights", reservation_id="ABC", flights=fl[:1]), "", rng)
    assert json.loads(c["function"]["arguments"])["flights"][0]["date"] == "2024-05-20"
    c, _ = s.corrupt(
        call("update_reservation_baggages", reservation_id="ABC", total_baggages=2, nonfree_baggages=0),
        "",
        rng,
    )
    assert json.loads(c["function"]["arguments"])["total_baggages"] == 3
    assert (
        s.corrupt(call("cancel_pending_order", order_id="#W1234567"), "#W1234567", rng) is None
    )  # no other order
