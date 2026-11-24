"""The hybrid judge (scripts/labels/hybrid_judge.py): the facts code computes, the checks on the
judge's answer, and the scoring. Hand-made conversations; no API calls."""

from __future__ import annotations

import json

from _scripts import load_script

hj = load_script("labels/hybrid_judge")


def call(name, i, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


PRODUCT = {
    "name": "Desk Lamp",
    "product_id": "p1",
    "variants": {
        "5320792178": {
            "item_id": "5320792178",
            "options": {"color": "black"},
            "available": True,
            "price": 135.24,
        },
        "1569765161": {
            "item_id": "1569765161",
            "options": {"color": "silver"},
            "available": True,
            "price": 143.02,
        },
    },
}
VIEW = [
    {"role": "system", "content": "POLICY"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Cancel FDZ0T5 and HSR97W please."},
    {"role": "assistant", "content": "", "tool_calls": [call("get_product_details", 0, product_id="p1")]},
    {"role": "tool", "tool_call_id": "c0", "content": json.dumps(PRODUCT)},
    {"role": "assistant", "content": "Cancel both? (yes/no)"},
    {"role": "user", "content": "Yes, cancel both."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("cancel_reservation", 1, reservation_id="FDZ0T5")],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "{}"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [call("cancel_reservation", 2, reservation_id="HSR97W")],
    },
    {"role": "tool", "tool_call_id": "c2", "content": "Error: not found"},
    {"role": "assistant", "content": "Both are cancelled."},
]


def test_every_change_is_listed_in_order_with_what_the_customer_said_before_it():
    made = hj.changes(VIEW, {"cancel_reservation"})
    assert [c["turn"] for c in made] == [7, 9]
    assert (
        made[0]["customer_before"] == "Yes, cancel both." and made[0]["accepted"] and not made[1]["accepted"]
    )


def test_prices_are_read_from_the_lookups_cheapest_first():
    rows = hj.prices(VIEW)
    assert rows[0].startswith("Desk Lamp | item 5320792178 | $135.24") and "$143.02" in rows[1]


def test_the_request_carries_the_facts_and_never_an_answer_key():
    msgs = hj.request(VIEW, "airline", [])
    user = msgs[1]["content"]
    assert "CHANGES MADE, IN ORDER" in user and "[9] cancel_reservation" in user and "REJECTED" in user
    assert "PRICES IN THE LOOKUPS" not in user  # retail only
    assert "expected" not in user.lower()


def test_an_error_counts_only_with_a_quote_and_an_agent_turn():
    ok = hj.verdict(
        {"error": True, "turn": 11, "kind": "wrong_information", "evidence": "Both are cancelled."}, VIEW
    )
    assert ok["flagged"] and ok["turn"] == 11
    made_up = hj.verdict({"error": True, "turn": 11, "evidence": "I refunded you $500"}, VIEW)
    assert not made_up["flagged"] and made_up["invalid"] == "evidence not in the conversation"
    customer = hj.verdict({"error": True, "turn": 6, "evidence": "Yes, cancel both."}, VIEW)
    assert not customer["flagged"] and customer["invalid"] == "turn isn't an agent turn"
    assert not hj.verdict({"error": False, "turn": 3}, VIEW)["flagged"]


def row(passed, true_turn, flagged, turn, domain="retail"):
    v = {"flagged": flagged, "turn": turn, "kind": None}
    return {
        "domain": domain,
        "passed": passed,
        "true_turn": true_turn,
        "rules": v,
        "checker": dict(v, flagged=False),
    }


def test_scoring_counts_recall_false_flags_and_localisation():
    rows = [
        row(False, 7, True, 7),
        row(False, 7, True, 15),
        row(False, None, False, None),
        row(True, None, True, 3),
    ]
    s = hj.score(rows, ["rules"])["rules"]["all"]
    assert s["failures_flagged"] == [2, 3] and s["passes_flagged"] == [1, 1]
    assert s["at_true_turn"] == [1, 2] and s["within_near"] == [1, 2]
    both = hj.combine({"rules": {"flagged": False, "turn": None}, "checker": {"flagged": True, "turn": 9}})
    assert both == {"flagged": True, "turn": 9, "kind": None}


def test_a_quote_from_the_computed_facts_counts_and_a_paraphrase_does_not():
    computed = hj.facts(VIEW, "airline", [])
    quote = '[9] cancel_reservation({"reservation_id": "HSR97W"}) | tool REJECTED'
    assert hj.verdict({"error": True, "turn": 11, "evidence": quote}, VIEW, computed)["flagged"]
    assert not hj.verdict({"error": True, "turn": 11, "evidence": quote}, VIEW)["flagged"]
    para = hj.verdict(
        {"error": True, "turn": 11, "evidence": "The agent never cancelled HSR97W properly"}, VIEW, computed
    )
    assert para["invalid"] == "evidence not in the conversation"


def test_kinds_count_the_answer_keys_and_the_judges_flags_among_failures():
    rows = [
        {
            "passed": False,
            "true_kind": "unexpected_change",
            "hybrid": {"flagged": True, "kind": "wrong_arguments"},
        },
        {"passed": False, "true_kind": "change_never_made", "hybrid": {"flagged": False, "kind": None}},
        {"passed": True, "true_kind": None, "hybrid": {"flagged": True, "kind": "stopped_short"}},
    ]
    k = hj.kinds(rows, ["rules", "hybrid"])
    assert k["answer_key"] == {"unexpected_change": 1, "change_never_made": 1}
    assert k["hybrid"] == {"wrong_arguments": 1}
