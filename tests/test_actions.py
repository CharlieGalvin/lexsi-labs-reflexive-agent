"""SEL-83: the check_action interface between the checker and the guard."""

from __future__ import annotations

from reflexive.actions import RULES, ActionFinding, check_action

CALL = {
    "id": "c1",
    "type": "function",
    "function": {"name": "cancel_pending_order", "arguments": '{"order_id": "#W1"}'},
}
VIEW = [{"role": "system", "content": "policy"}, {"role": "user", "content": "Cancel order #W1."}]


def test_with_no_rules_nothing_is_flagged_and_sel_88s_rule_is_the_default():
    assert check_action(VIEW, CALL, "policy", rules=[]) is None
    from reflexive.intent import no_yes

    assert RULES == [no_yes]  # SEL-88; the hand-written mismatch rule is opt-in
    f = check_action(VIEW, CALL, "policy")  # a request up front, never described or confirmed
    assert f is not None and (f.rule, f.live) == ("no_yes", True)


def test_a_rule_gets_the_conversation_the_call_and_the_policy_and_live_findings_come_first():
    seen = []

    def later(conversation, call, policy):
        seen.append((len(conversation), call["function"]["name"], policy))
        return ActionFinding("INTENT", "later", "only after the fact", live=False)

    def live(conversation, call, policy):
        return ActionFinding("INTENT", "no_yes", "no yes from the customer", live=True)

    first = check_action(VIEW, CALL, "policy", rules=[later, live])
    assert first is not None and first.rule == "no_yes"
    only_later = check_action(VIEW, CALL, "policy", rules=[later])
    assert only_later is not None and only_later.live is False
    assert seen[0] == (2, "cancel_pending_order", "policy")
    assert check_action(VIEW, CALL, "policy", rules=[lambda *a: None]) is None


def test_an_after_the_fact_finding_is_kept_on_the_live_finding_that_wins():
    def judge_error(conversation, call, policy):
        return ActionFinding("INTENT", "no_yes_judge_error", "the judge couldn't answer", live=False)

    def meta(conversation, call, policy):
        return ActionFinding("META", "wrong_item", "wrong item", live=True)

    f = check_action(VIEW, CALL, "policy", rules=[judge_error, meta])
    assert f is not None and f.rule == "wrong_item" and [x.rule for x in f.also] == ["no_yes_judge_error"]
    assert f.to_dict()["also"][0]["rule"] == "no_yes_judge_error"  # what the guard's log records
    alone = check_action(VIEW, CALL, "policy", rules=[meta])
    assert alone is not None and alone.also == ()
