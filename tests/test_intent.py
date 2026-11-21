"""SEL-88: the "no yes" rule, made-up intent (reflexive.intent)."""

from __future__ import annotations

import json

import pytest

from reflexive.actions import RULES, check_action
from reflexive.intent import WRITE_TOOLS, describes, intent_findings, is_yes, no_yes

SYS = {"role": "system", "content": "policy text"}


def agent(text: str) -> dict:
    return {"role": "assistant", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def call(name: str, **args) -> dict:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


CANCEL = call("cancel_pending_order", order_id="#W2378156", reason="no longer needed")
DESCRIBE = agent("I can cancel order #W2378156 (reason: no longer needed). Shall I go ahead?")


def flagged(conversation, c=CANCEL) -> bool:
    return no_yes([SYS, *conversation], c, "policy text") is not None


def test_the_rule_is_registered_with_check_action_and_is_live():
    assert no_yes in RULES
    f = check_action([SYS, user("Please cancel order #W2378156.")], CANCEL, "policy text")
    assert f is not None and (f.type, f.rule, f.live) == ("INTENT", "no_yes", True)
    assert "cancel_pending_order on #W2378156" in f.reason


def test_a_request_up_front_is_not_a_yes_to_the_agents_later_proposal():
    assert flagged([user("Cancel order #W2378156 please, I no longer need it.")])
    assert flagged([user("Cancel order #W2378156 please."), DESCRIBE])  # described, no answer yet


def test_a_yes_after_the_description_lets_the_change_run():
    assert not flagged([user("Cancel my order #W2378156."), DESCRIBE, user("Yes, please go ahead.")])
    for yes in ("yes", "Sure!", "OK", "That's fine.", "Please do.", "I confirm.", "No problem, go ahead."):
        assert not flagged([DESCRIBE, user(yes)]), yes


def test_a_yes_before_the_description_does_not_count():
    assert flagged([user("Yes, I want to cancel."), DESCRIBE])


def test_a_refusal_or_a_negation_is_not_a_yes_and_withdraws_an_earlier_one():
    for no in ("No, wait.", "Hold on, not yet.", "Don't cancel it.", "That's not correct.", "I'm not sure."):
        assert not is_yes(user(no)), no
    assert flagged([DESCRIBE, user("Yes."), user("Wait, actually no.")])
    assert not flagged([DESCRIBE, user("Yes."), user("Wait, actually no."), DESCRIBE, user("OK, yes now.")])


def test_a_description_of_another_order_is_a_different_change():
    other = agent("I can cancel order #W9999999 for you. Shall I go ahead?")
    assert flagged([other, user("Yes.")])


def test_one_yes_to_a_list_covers_every_change_on_it():
    both = agent("I'll cancel orders #W2378156 and #W1111111. Do you confirm both?")
    second = call("cancel_pending_order", order_id="#W1111111", reason="no longer needed")
    conv = [both, user("Yes, both please.")]
    assert not flagged(conv) and not flagged(conv, second)


def test_a_description_with_no_id_at_all_counts():
    assert not flagged([agent("I'll cancel that order for you now. Is that OK?"), user("Yes")])


def test_the_calls_other_ids_and_units_are_understood():
    flights = call(
        "update_reservation_flights",
        reservation_id="ZFA04Y",
        cabin="economy",
        flights=[{"flight_number": "HAT006", "date": "2024-05-20"}],
        payment_id="credit_card_4421486",
    )
    assert not flagged(
        [agent("I can change your flight to HAT006 on 20 May. Shall I proceed?"), user("Yes")], flights
    )
    refuel = call("refuel_data", customer_id="C1001", line_id="L1002", gb_amount=2.0)
    assert not flagged([agent("I can add 2 GB of data (10GB plan top-up) for $5. OK?"), user("Sure")], refuel)
    assert not describes(
        agent("I can add data to line L9999. OK?"), "refuel_data", json.loads(refuel["function"]["arguments"])
    )


def test_only_state_changing_calls_are_checked():
    assert no_yes([SYS], call("get_order_details", order_id="#W2378156"), "") is None


def test_intent_findings_checks_each_call_against_the_conversation_before_it():
    rec = {
        "domain": "retail",
        "agent_view": [
            SYS,
            user("Cancel #W2378156 please."),
            {"role": "assistant", "content": "", "tool_calls": [CANCEL]},  # before any yes: flagged
            {"role": "tool", "content": "error", "tool_call_id": "c1"},
            DESCRIBE,
            user("Yes."),
            {"role": "assistant", "content": "", "tool_calls": [CANCEL]},  # after the yes: fine
        ],
    }
    (f,) = intent_findings(rec)
    assert (f["turn"], f["tool"], f["rule"]) == (2, "cancel_pending_order", "no_yes")


def test_it_reads_no_answer_key():
    rec = {
        "domain": "retail",
        "agent_view": [SYS, user("Cancel it."), {"role": "assistant", "tool_calls": [CANCEL]}],
    }
    with_key = {**rec, "task": {"evaluation_criteria": {"actions": [{"name": "cancel_pending_order"}]}}}
    with_key["outcome"] = {"passed": True, "reward": 1.0}
    assert intent_findings(rec) == intent_findings(with_key)


def test_the_write_tools_match_tau2_bench():
    pytest.importorskip("tau2")
    from tau2.environment.toolkit import ToolType
    from tau2.registry import registry

    for domain, tools in WRITE_TOOLS.items():
        tk = registry.get_env_constructor(domain)().tools
        assert tk is not None
        live = {n for n in tk.get_tools() if tk.tool_type(n) == ToolType.WRITE}
        assert live == tools, domain
