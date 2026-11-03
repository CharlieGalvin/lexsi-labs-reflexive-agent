"""reflexive.teacher: the teacher model as a Tau2Episode agent (SEL-37). No API calls."""

from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from reflexive.teacher import (
    TOOL_RESULT,
    RateLimiter,
    TeacherClient,
    parse_prompted,
    to_prompted,
)

TOOLS = [{"type": "function", "function": {"name": "get_order_details", "parameters": {}}}]


def fake_completion(replies):
    """A stand-in for litellm.completion that returns the given replies in order."""
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        r = replies[len(calls) - 1]
        if isinstance(r, str):
            msg = NS(content=r, tool_calls=None)
        else:  # native tool calls
            msg = NS(content=None, tool_calls=[
                NS(id=c["id"], function=NS(name=c["name"], arguments=json.dumps(c["args"])))
                for c in r])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=1000, completion_tokens=50))

    completion.calls = calls
    return completion


# --- Prompted format ----------------------------------------------------------------------------


def test_to_prompted_puts_tools_in_system_and_folds_results():
    convo = [
        {"role": "system", "content": "POLICY"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Where's my order?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_order_details", "arguments": '{"order_id": "#W1"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"status":"pending"}'},
    ]
    out = to_prompted(convo, TOOLS)
    assert out[0]["role"] == "system" and out[0]["content"].startswith("POLICY")
    assert "get_order_details" in out[0]["content"] and '"tool_name"' in out[0]["content"]
    assert json.loads(out[3]["content"]) == [
        {"tool_name": "get_order_details", "parameters": {"order_id": "#W1"}}]
    assert out[4] == {"role": "user", "content": TOOL_RESULT.format(
        name="get_order_details", content='{"status":"pending"}')}
    assert all(m["role"] != "tool" for m in out)


def test_to_prompted_merges_adjacent_turns_from_the_same_role():
    convo = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "a", "function": {"name": "x", "arguments": {}}},
        {"id": "b", "function": {"name": "y", "arguments": {}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"}]
    out = to_prompted(convo, TOOLS)
    assert [m["role"] for m in out] == ["assistant", "user"]
    assert "Tool result for x: 1" in out[1]["content"] and "Tool result for y: 2" in out[1]["content"]


@pytest.mark.parametrize("text,expected", [
    ('[{"tool_name": "get_order_details", "parameters": {"order_id": "#W1"}}]',
     [("get_order_details", {"order_id": "#W1"})]),
    ('{"tool_name": "get_order_details", "parameters": {}}', [("get_order_details", {})]),
    ('```json\n[{"name": "get_order_details", "arguments": {"order_id": "#W2"}}]\n```',
     [("get_order_details", {"order_id": "#W2"})]),
    ('[{"tool_name": "made_up_tool", "parameters": {}}]', [("made_up_tool", {})]),  # kept
])
def test_parse_prompted_reads_tool_calls(text, expected):
    calls = parse_prompted(text)
    assert [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls] == expected


@pytest.mark.parametrize("text", [
    "Your order #W1 is pending.",
    'I can look that up. Send me {"order_id": ...} please.',  # mentions JSON, doesn't start with it
    "[not json",
    '[{"parameters": {}}]',  # no tool name
])
def test_parse_prompted_leaves_messages_alone(text):
    assert parse_prompted(text) is None


# --- Client -------------------------------------------------------------------------------------


def test_prompted_turn_returns_tool_calls_and_counts_usage():
    fc = fake_completion(['[{"tool_name": "get_order_details", "parameters": {"order_id": "#W1"}}]',
                          "Your order is pending."])
    t = TeacherClient("prompted", completion=fc, rpm=0)
    turn = t.turn([{"role": "system", "content": "P"}, {"role": "user", "content": "hi"}], TOOLS)
    assert turn["content"] is None and turn["tool_calls"][0]["function"]["name"] == "get_order_details"
    assert "tools" not in fc.calls[0]  # prompted: no native tools parameter
    assert fc.calls[0]["model"] == "openai/c4ai-aya-expanse-32b"
    assert fc.calls[0]["api_base"] == "https://api.cohere.ai/compatibility/v1"
    msg = t.turn([{"role": "user", "content": "and?"}], TOOLS)
    assert msg == {"role": "assistant", "content": "Your order is pending."}
    assert t.usage.calls == 2 and t.usage.prompt_tokens == 2000
    assert t.usage.cost == pytest.approx(2000 * 0.5e-6 + 100 * 1.5e-6)


def test_native_turn_passes_tools_and_reads_structured_calls():
    fc = fake_completion([[{"id": "x1", "name": "get_order_details", "args": {"order_id": "#W1"}}]])
    t = TeacherClient("native", completion=fc, rpm=0)
    turn = t.turn([{"role": "user", "content": "hi"}], TOOLS)
    assert fc.calls[0]["tools"] == TOOLS
    assert turn["tool_calls"][0]["id"] == "x1"


def test_unparsable_json_reply_is_sent_as_a_message_and_counted():
    t = TeacherClient("prompted", completion=fake_completion(['[{"tool_name": "x", "parameters": ']),
                      rpm=0)
    turn = t.turn([{"role": "user", "content": "hi"}], TOOLS)
    assert "tool_calls" not in turn and t.usage.unparsed == 1


def test_rate_limiter_spaces_calls():
    now, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s

    rl = RateLimiter(20, clock=lambda: now[0], sleep=sleep)
    for _ in range(3):
        rl.wait()
    assert slept == [pytest.approx(3.0), pytest.approx(3.0)]  # 60 / 20 = 3 s apart


def test_no_rate_limit_by_default(monkeypatch):
    monkeypatch.delenv("TEACHER_RPM", raising=False)
    t = TeacherClient(completion=fake_completion([]))
    assert t.limiter.interval == 0.0
    monkeypatch.setenv("TEACHER_RPM", "20")
    assert TeacherClient(completion=fake_completion([])).limiter.interval == pytest.approx(3.0)


def test_missing_key_is_explained(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="COHERE_API_KEY"):
        TeacherClient()


# --- End to end through Tau2Episode (needs τ²-bench) -------------------------------------------


def test_prompted_teacher_solves_a_task_through_the_episode():
    pytest.importorskip("tau2")
    from loguru import logger

    from reflexive.episode import Tau2Episode
    from reflexive.outcome import scorer

    logger.disable("tau2")
    gold = [m for m in scorer("retail").gold_trajectory("0") if m["role"] == "assistant"]
    replies = [json.dumps([{"tool_name": tc["name"], "parameters": tc["arguments"]}
                           for tc in m["tool_calls"]]) for m in gold]
    replies.append("Your exchange is confirmed. ###STOP###")  # τ²-bench's agent stop
    teacher = TeacherClient("prompted", completion=fake_completion(replies), rpm=0)
    ep = Tau2Episode("retail", "0", user_llm="gpt-4.1",
                     user_llm_args={"mock_response": "Yes, please go ahead."},
                     max_context=128_000, reserve=4_096)
    ep.reset()
    while not ep.done:
        ep.step(teacher.turn(ep.prompt(), ep.tool_schemas))
    rec = ep.record(judges=False)
    assert rec["termination"] == "agent_stop" and rec["outcome"]["passed"]
    assert teacher.usage.calls == len(replies)
    names = [tc["function"]["name"] for m in rec["messages"] for tc in m.get("tool_calls") or []]
    assert names == [tc["name"] for m in gold for tc in m["tool_calls"]]
