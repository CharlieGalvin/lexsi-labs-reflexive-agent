"""SEL-83: the check_action interface between the checker and the guard."""

from __future__ import annotations

from reflexive.actions import RULES, ActionFinding, check_action

CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "#W1"}'},
}
VIEW = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Cancel order #W1."}]


def test_with_no_rules_nothing_is_flagged():
    assert RULES == []  # SEL-88 adds the first rule
    assert check_action(VIEW, CALL, "policy") is None


def test_a_rule_gets_the_conversation_the_call_and_the_policy_and_live_findings_come_first():
    seen = []

    def later(conversation, call, policy):
        seen.append((len(conversation), call["function"]["name"], policy))
        return ActionFinding("INTENT", "later", "only after the fact", live=False)

    def live(conversation, call, policy):
        return ActionFinding("INTENT", "no_yes", "no yes from the customer", live=True)

    assert check_action(VIEW, CALL, "policy", rules=[later, live]).rule == "no_yes"
    assert check_action(VIEW, CALL, "policy", rules=[later]).live is False
    assert seen[0] == (2, "cancel_pending_order", "policy")
    assert check_action(VIEW, CALL, "policy", rules=[lambda *a: None]) is None
