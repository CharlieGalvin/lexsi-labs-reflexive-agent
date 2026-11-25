"""Full-conversation fixes (scripts/heal/continue_fixes.py): cutting a saved conversation at the fix,
and which turns are trained on. The τ²-bench part is a stand-in episode: no API calls."""

from __future__ import annotations

import json

import pytest
from _scripts import load_script

from reflexive.splits import task_ids

c = load_script("heal/continue_fixes")
TASK = sorted(task_ids("retail", "train"))[3]


def call(name, i=0, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


MESSAGES = [
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "Cancel #W1."},
    {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", 0, order_id="#W1")]},
    {"role": "tool", "tool_call_id": "c0", "content": "{}"},
    {"role": "assistant", "content": "", "tool_calls": [call("cancel_pending_order", 1, order_id="#W1")]},
    {"role": "tool", "tool_call_id": "c1", "content": "{}"},
]
REC = {
    "domain": "retail",
    "task_id": TASK,
    "sample": 0,
    "user_llm": "gpt-4.1-2025-04-14",
    "agent_view": [{"role": "system", "content": "SYSTEM"}, *MESSAGES],
    "messages": MESSAGES,
}


def test_a_fix_id_is_read_back_including_a_task_id_with_slashes():
    info = c.parse_id("mismatch/glm53f/telecom/a/b[c]/S2-telecom-fresh/s1/t21/a2")
    assert info == {
        "teacher": "glm53f",
        "domain": "telecom",
        "task_id": "a/b[c]",
        "source": "S2-telecom-fresh",
        "sample": 1,
        "turn": 21,
    }
    with pytest.raises(ValueError):
        c.parse_id("onpolicy/glm53f/retail/1/s0/t2/a1")


def test_the_conversation_is_cut_just_before_the_flagged_turn():
    history, k = c.cut(REC, 5)  # view turn 5 is the cancel (view[0] is the system prompt)
    assert history == MESSAGES[:4] and k == 2
    with pytest.raises(ValueError):
        c.cut(REC, 4)  # a tool message, not an agent turn


def test_only_the_fixed_turn_and_after_are_trained_on():
    msgs = [
        {"role": r, "content": ""} for r in ("system", "assistant", "user", "assistant", "user", "assistant")
    ]
    assert c.train_from(msgs, 1) == [3, 5] and c.train_from(msgs, 0) == [1, 3, 5]
    with pytest.raises(ValueError):
        c.train_from(msgs, 3)


class Episode:
    """A stand-in for Tau2Episode: the customer says thanks after each agent turn, then stops."""

    def __init__(self, domain, task_id, *, user_llm, prompt_variant, history):
        self.history, self.sent, self.done = history, [], False
        self.tool_schemas, self.passed = [], True

    def reset(self):
        return []

    def step(self, reply):
        self.sent.append(reply)
        if len(self.sent) == 3:
            self.done = True
            return []
        thanks = {"role": "user", "content": "Yes, thanks."}
        self.sent.append(thanks)
        return [thanks]

    def prompt(self):
        return [{"role": "system", "content": "SYSTEM"}, *self.history, *self.sent]

    def record(self, judges, **extra):
        msgs = [*self.history, *self.sent]
        return {
            "domain": "retail",
            "task_id": TASK,
            "messages": msgs,
            "termination": "agent_stop",
            "outcome": {"passed": self.passed, "reward": 1.0, "checks": {}},
            **extra,
        }


class Teacher:
    def __init__(self):
        self.seen = []

    def turn(self, messages, tools):
        self.seen.append(messages)
        return {"role": "assistant", "content": "Done. ###STOP###"}


FIX = {
    "id": f"mismatch/glm53f/retail/{TASK}/S2-retail-fresh/s0/t5/a1",
    "source": {"fixed_turn_message": {"role": "assistant", "content": "Just to check: cancel #W1, right?"}},
}


def test_the_fixed_turn_goes_first_then_the_teacher_carries_on_and_a_failure_is_not_kept(monkeypatch):
    made = []

    def episode(*a, **kw):
        made.append(Episode(*a, **kw))
        made[-1].passed = False
        return made[-1]

    teacher = Teacher()
    ex, run, why = c.continue_one(FIX, REC, lambda: teacher, episode=episode)
    (ep,) = made
    assert ep.history == MESSAGES[:4]  # resumed just before the flagged cancel
    assert ep.sent[0]["content"] == "Just to check: cancel #W1, right?" and ep.sent[2]["content"].startswith(
        "Done"
    )
    assert "Just to check" in str(teacher.seen[0])  # the teacher sees its fixed turn in the conversation
    assert ex is None and why == "the whole conversation didn't pass" and run["fix_id"] == FIX["id"]


def test_a_passing_continuation_trains_on_the_fixed_turn_and_after_only():
    ex, run, why = c.continue_one(FIX, REC, Teacher, episode=Episode)
    assert why == "kept" and ex["group"] == "mismatch" and ex["id"].startswith("mismatch-full/")
    agent = [i for i, m in enumerate(ex["messages"]) if m["role"] == "assistant"]
    assert ex["train_on"] == agent[2:]  # two agent turns before the fix are the student's own
    assert "Just to check" in ex["messages"][ex["train_on"][0]]["content"]
    assert ex["source"]["fixed_agent_turn"] == 2 and ex["source"]["agent_turns_trained"] == 2
