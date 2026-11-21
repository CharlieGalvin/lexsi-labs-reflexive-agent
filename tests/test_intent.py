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


# --- The LLM judge behind the word match ------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from reflexive.intent import no_yes_words  # noqa: E402
from reflexive.intent_judge import Judge, quote_found, request  # noqa: E402

ASKED = [SYS, DESCRIBE, user("Both were ordered by mistake. Just cancel them, please.")]


def fake(answer):
    """A stand-in for litellm.completion that returns ``answer`` and records each request."""
    seen = []

    def completion(**kw):
        seen.append(kw)
        content = answer if isinstance(answer, str) else json.dumps(answer)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    completion.seen = seen  # ty: ignore[unresolved-attribute]
    return completion


def test_an_instruction_is_missed_by_the_words_and_found_by_the_judge():
    assert no_yes_words(ASKED, CANCEL, "") is not None  # "Just cancel them, please." has no yes word
    j = Judge(
        completion=fake({"yes_given": True, "quote": "Just cancel them, please", "reason": "instruction"})
    )
    assert no_yes(ASKED, CANCEL, "", judge=j) is None
    kw = j._completion.seen[0]  # ty: ignore[unresolved-attribute]
    assert kw["model"] == "gpt-4.1-mini-2025-04-14" and kw["temperature"] == 0


def test_the_judge_is_only_asked_when_the_words_would_flag():
    j = Judge(completion=fake({"yes_given": False, "quote": "", "reason": "x"}))
    assert no_yes([SYS, DESCRIBE, user("Yes please.")], CANCEL, "", judge=j) is None
    assert j.calls == 0


def test_a_yes_whose_quote_isnt_the_customers_words_does_not_count():
    j = Judge(completion=fake({"yes_given": True, "quote": "Yes, cancel it", "reason": "made up"}))
    f = no_yes(ASKED, CANCEL, "", judge=j)
    assert f is not None and f.live and "quote isn't in any customer message" in f.reason
    assert quote_found("just cancel them please", ASKED) and not quote_found("", ASKED)


def test_no_yes_from_the_judge_flags_live():
    j = Judge(completion=fake({"yes_given": False, "quote": "", "reason": "a question, then the change"}))
    f = no_yes([SYS, user("Can I return the backpack too?")], CANCEL, "", judge=j)
    assert f is not None and (f.rule, f.live) == ("no_yes", True) and "a question" in f.reason


def test_a_judge_error_never_blocks_and_is_logged():
    def broken(**kw):
        raise TimeoutError("slow")

    f = no_yes(ASKED, CANCEL, "", judge=Judge(completion=broken))
    assert f is not None and (f.rule, f.live) == ("no_yes_judge_error", False)
    bad_json = no_yes(ASKED, CANCEL, "", judge=Judge(completion=fake("not json")))
    assert bad_json is not None and bad_json.live is False


def test_answers_are_cached_by_the_exact_request(tmp_path):
    path = tmp_path / "cache.jsonl"
    answer = {"yes_given": True, "quote": "Just cancel them, please", "reason": "instruction"}
    j = Judge(completion=fake(answer), cache_path=path)
    assert j(ASKED, CANCEL).yes_given and j(ASKED, CANCEL).cached and j.calls == 1
    again = Judge(completion=fake({"yes_given": False}), cache_path=path)  # a new run reads the file
    assert again(ASKED, CANCEL).yes_given and again.calls == 0


def test_the_judge_sees_the_conversation_and_the_change_but_no_system_prompt_or_tool_results():
    conv = [SYS, DESCRIBE, {"role": "tool", "content": "SECRET RESULT"}, user("ok then")]
    text = request(conv, CANCEL)[1]["content"]
    assert "policy text" not in text and "SECRET RESULT" not in text
    assert "CUSTOMER: ok then" in text and "cancel_pending_order" in text


def test_tests_run_with_words_only():
    import os

    assert os.environ["REFLEXIVE_INTENT_JUDGE"] == "off"
    assert no_yes(ASKED, CANCEL, "") is not None  # no network: the default judge is off
