"""Steps for features/agent_view.feature: what the agent sees, for training and the classifier.

The first two scenarios use the hand-made telecom conversation from test_golden.py (offline);
the last plays a real τ²-bench task and is skipped without τ²-bench.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pytest_bdd import given, scenarios, then, when
from test_golden import SP, record

from reflexive import golden
from reflexive.hallucination import agent_view_of

scenarios("features/agent_view.feature")

Message = dict[str, Any]


@dataclass
class Conversation:
    """A conversation and the view of it under test."""

    messages: list[Message]
    view: list[Message] = field(default_factory=list)
    calls_made: int = 0
    done: bool = False

    def customer_calls(self) -> set[str]:
        return {tc["id"] for m in self.messages if m["role"] == "user" for tc in m.get("tool_calls") or []}


@given("a telecom conversation in which the customer checked their own phone", target_fixture="conv")
def telecom_conversation() -> Conversation:
    conv = Conversation(record()["messages"])
    assert conv.customer_calls(), "the example must include a customer phone action"
    return conv


@when("the agent's view is rebuilt from the conversation")
def rebuild_for_training(conv: Conversation) -> None:
    conv.view = golden.agent_view(conv.messages, SP)


@when("the classifier rebuilds what the agent saw")
def rebuild_for_classifier(conv: Conversation) -> None:
    conv.view = agent_view_of({"messages": conv.messages})


@then("it has no customer tool calls and none of their results")
def no_customer_actions(conv: Conversation) -> None:
    hidden = conv.customer_calls()
    assert not any(m["role"] == "user" and m.get("tool_calls") for m in conv.view)
    assert not any(m["role"] == "tool" and m.get("tool_call_id") in hidden for m in conv.view)


@then("it still has every message the customer wrote")
def customer_words_kept(conv: Conversation) -> None:
    def said(msgs: list[Message]) -> list[str]:
        return [m["content"] for m in msgs if m["role"] == "user" and m.get("content")]

    assert said(conv.view) == said(conv.messages)


@then("it still has the agent's own tool calls and their results")
def agent_calls_kept(conv: Conversation) -> None:
    agent_ids = {
        tc["id"] for m in conv.messages if m["role"] == "assistant" for tc in m.get("tool_calls") or []
    }
    seen_calls = {tc["id"] for m in conv.view if m["role"] == "assistant" for tc in m.get("tool_calls") or []}
    seen_results = {m.get("tool_call_id") for m in conv.view if m["role"] == "tool"}
    assert agent_ids and agent_ids == seen_calls == seen_results


# --- A real conversation through our wrapper (needs τ²-bench) -------------------------------------


@given(
    "a retail conversation in which the agent made the task's correct calls, then ended it",
    target_fixture="conv",
)
def finished_retail_conversation() -> Conversation:
    pytest.importorskip("tau2")
    from test_episode import play_correct_actions  # skips the module without τ²-bench's data

    ep, _, seen = play_correct_actions("retail", "0", max_context=200_000)
    return Conversation([], view=ep.messages(), calls_made=len(seen), done=ep.done)


@then("the conversation is over")
def is_over(conv: Conversation) -> None:
    assert conv.done


@then("the saved view still starts with the system prompt")
def starts_with_system(conv: Conversation) -> None:
    assert conv.view and conv.view[0]["role"] == "system"


@then("it still has the result of every call the agent made")
def every_result(conv: Conversation) -> None:
    assert conv.calls_made > 0 and sum(m["role"] == "tool" for m in conv.view) == conv.calls_made
