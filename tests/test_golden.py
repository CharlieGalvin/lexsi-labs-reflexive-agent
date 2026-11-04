"""Golden trajectories v1 (SEL-9): the agent's view, picking, Cohere conversion and its checks.

Offline: a small hand-made conversation stands in for a teacher record.
"""

import copy
import json

import pytest

from reflexive import golden

SP = "<instructions>be helpful</instructions>"
TOOLS = [{"type": "function", "function": {
    "name": "get_user", "description": "Look up a user.",
    "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}},
                   "required": ["user_id"]}}}]


def call(i, name, args):
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def record(**over):
    msgs = [
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "My phone has no signal. I'm user u_123."},
        {"role": "assistant", "content": None, "tool_calls": [call("a1", "get_user", {"user_id": "u_123"})]},
        {"role": "tool", "tool_call_id": "a1", "content": '{"name": "Ann",  "line": "L1"}'},
        {"role": "assistant", "content": "Thanks Ann. Please check your status bar."},
        {"role": "user", "content": None, "tool_calls": [call("c1", "check_status_bar", {})]},
        {"role": "tool", "tool_call_id": "c1", "content": "No Signal"},
        {"role": "user", "content": "It says No Signal."},
        {"role": "assistant", "content": "Please restart your phone. ###STOP###"},
    ]
    r = {"domain": "telecom", "task_id": "t1", "attempt": 1, "termination": "agent_stop",
         "outcome": {"passed": True, "reward": 1.0, "checks": {"ENV_ASSERTION": 1.0}, "skipped": []},
         "messages": msgs, "sample_model": "glm53f", "temperature": 0.0,
         "agent_usage": {"providers": {"Wafer": 3}}, "user_llm": "openrouter/openai/gpt-4.1"}
    r.update(over)
    return r


def test_agent_view_drops_customer_phone_actions_and_compacts():
    view = golden.agent_view(record()["messages"], SP)
    assert view[0] == {"role": "system", "content": SP}
    assert not any(m["role"] == "user" and m.get("tool_calls") for m in view)
    assert [m["content"] for m in view if m["role"] == "tool"] == ['{"name":"Ann","line":"L1"}']
    assert len(view) == 1 + 9 - 2


def test_to_record_fields_and_targets():
    rec = golden.to_record(record(), SP, TOOLS, "train")
    assert rec["id"] == "glm53f/telecom/t1/1" and rec["format"] == "reagent.golden/v1"
    assert rec["teacher"] == {"model": "z-ai/glm-5.3-flash", "provider": "Wafer",
                              "temperature": 0.0, "attempt": 1}
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "assistant", "user", "assistant", "user", "assistant", "user", "assistant"]
    assert rec["train_on"] == [1, 3, 5, 7]
    assert rec["messages"][3]["content"].startswith("Action: ```json")
    assert rec["messages"][4]["content"].startswith("<results>\nDocument: 0\ntool_name: get_user")
    assert "## Available Tools" in rec["messages"][0]["content"]
    assert "check_status_bar" not in json.dumps(rec["messages"])
    assert golden.check_record(rec, golden.agent_view(record()["messages"], SP)) == []


def test_never_a_test_record():
    with pytest.raises(ValueError):
        golden.to_record(record(), SP, TOOLS, "test")


@pytest.mark.parametrize("break_it, expect", [
    (lambda r: r["messages"][3].update(content=r["messages"][3]["content"].replace("u_123", "u_999")),
     "round trip"),
    (lambda r: r["messages"][5].update(content="Thanks. Please check your status bar."), "round trip"),
    (lambda r: r["messages"][4].update(content=r["messages"][4]["content"].replace("Ann", "Bob")),
     "round trip"),
    (lambda r: r.update(train_on=[3, 5, 7]), "train_on"),
    (lambda r: r["messages"].insert(2, {"role": "user", "content": "extra"}), "alternate"),
    (lambda r: r["messages"][0].update(content="other prompt"), "system prompt"),
    (lambda r: r["outcome"].update(reward=0.0), "reward"),
])
def test_check_record_catches(break_it, expect):
    rec = golden.to_record(record(), SP, TOOLS, "train")
    bad = copy.deepcopy(rec)
    break_it(bad)
    assert any(expect in p for p in golden.check_record(bad, golden.agent_view(record()["messages"], SP)))


def test_merged_turns_are_caught():
    """A tool result followed by customer text would merge into one user turn: not a clean read-back."""
    r = record()
    r["messages"].insert(4, {"role": "user", "content": "Also my email changed."})
    rec = golden.to_record(r, SP, TOOLS, "train")
    assert golden.check_record(rec, golden.agent_view(r["messages"], SP))


def test_pick_first_clean_pass():
    fail = record(attempt=1, outcome={"passed": False, "reward": 0.0, "checks": {}, "skipped": []})
    retry = record(attempt=2)
    retry["messages"] = retry["messages"][:4] + [
        {"role": "assistant", "content": None, "tool_calls": [call("a2", "get_user", {"user_id": "x"})]},
        {"role": "tool", "tool_call_id": "a2", "content": "Error: not found"},
        {"role": "assistant", "content": None, "tool_calls": [call("a3", "get_user", {"user_id": "x"})]},
        {"role": "tool", "tool_call_id": "a3", "content": "Error: not found"},
    ] + retry["messages"][4:]
    good = record(attempt=3)
    later = record(attempt=4)
    assert golden.why_not_clean(fail) == "failed a check"
    assert golden.why_not_clean(retry) == "retried a failed call"
    assert golden.why_not_clean(record(termination="max_steps")).startswith("run broke")
    assert golden.why_not_clean(record(error="boom")).startswith("run broke")
    assert golden.pick([later, retry, fail, good])[("telecom", "t1")]["attempt"] == 3
