"""Steps for features/golden_trajectories.feature: teacher conversation → training record (SEL-9).

Uses the small hand-made telecom conversation from test_golden.py, so it runs without τ²-bench.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from test_golden import SP, TOOLS, record

from reflexive import golden

scenarios("features/golden_trajectories.feature")

Record = dict[str, Any]


@dataclass
class Case:
    """The conversation under test and everything derived from it in a scenario."""

    conversation: Record
    golden: Record | None = None
    error: Exception | None = None
    attempts: list[Record] = field(default_factory=list)
    kept: Record | None = None

    def view(self) -> list[Record]:
        return golden.agent_view(self.conversation["messages"], SP)


@given(
    "a telecom conversation the teacher solved, in which the customer checked their phone",
    target_fixture="case",
)
def solved_conversation() -> Case:
    conv = record()
    assert any(m["role"] == "user" and m.get("tool_calls") for m in conv["messages"])  # a phone action
    return Case(conv)


@when("it is turned into a training record")
def convert(case: Case) -> None:
    case.golden = golden.to_record(case.conversation, SP, TOOLS, "train")


@when("it is turned into a training record for the test split")
def convert_test(case: Case) -> None:
    try:
        golden.to_record(case.conversation, SP, TOOLS, "test")
    except ValueError as e:
        case.error = e


def _rec(case: Case) -> Record:
    assert case.golden is not None, "no training record was made"
    return case.golden


@then("the customer's own phone actions are not in the record")
def no_phone_actions(case: Case) -> None:
    msgs = _rec(case)["messages"]
    assert "check_status_bar" not in json.dumps(msgs)  # the customer's phone action
    results = [m["content"] for m in msgs[1:] if m["content"].startswith("<results>")]
    assert results and all("get_user" in r for r in results)  # only the agent's own tool result


@then("the record starts with the system prompt, which lists the agent's tools")
def system_first(case: Case) -> None:
    first = _rec(case)["messages"][0]
    assert first["role"] == "system" and first["content"].startswith(SP)
    assert "## Available Tools" in first["content"] and "def get_user(" in first["content"]


@then('each agent tool call is written as an "Action:" block')
def action_blocks(case: Case) -> None:
    msgs = _rec(case)["messages"]
    calls = [m for m in msgs if m["role"] == "assistant" and "tool_name" in m["content"]]
    assert calls and all(m["content"].startswith("Action: ```json") for m in calls)


@then("each tool result comes back as a customer-side turn inside <results> tags")
def results_tags(case: Case) -> None:
    msgs = _rec(case)["messages"][1:]  # the system prompt itself mentions <results> tags
    results = [m for m in msgs if "<results>" in m["content"]]
    assert results and all(m["role"] == "user" and m["content"].startswith("<results>") for m in results)


@then("every assistant turn is a learning target")
def all_assistant_targets(case: Case) -> None:
    rec = _rec(case)
    assert {i for i, m in enumerate(rec["messages"]) if m["role"] == "assistant"} <= set(rec["train_on"])


@then("no system, customer or tool-result turn is a learning target")
def nothing_else_targeted(case: Case) -> None:
    rec = _rec(case)
    assert all(rec["messages"][i]["role"] == "assistant" for i in rec["train_on"])


def _change_argument(r: Record) -> None:
    m = next(m for m in r["messages"] if m["content"].startswith("Action:"))
    m["content"] = m["content"].replace("u_123", "u_999")


def _change_message(r: Record) -> None:
    m = next(
        m for m in r["messages"][2:] if m["role"] == "assistant" and not m["content"].startswith("Action:")
    )
    m["content"] += " (edited)"


def _change_result(r: Record) -> None:
    m = next(m for m in r["messages"] if m["content"].startswith("<results>"))
    m["content"] = m["content"].replace("Ann", "Bob")


DAMAGE: dict[str, Callable[[Record], None]] = {
    "tool call argument": _change_argument,
    "agent message": _change_message,
    "tool result": _change_result,
    "list of learning targets": lambda r: r.update(train_on=r["train_on"][1:]),
    "system prompt": lambda r: r["messages"][0].update(content="a different prompt"),
}


@when(parsers.parse("the record's {part} is changed"))
def damage(case: Case, part: str) -> None:
    case.golden = copy.deepcopy(_rec(case))
    DAMAGE[part](case.golden)


@then(parsers.parse('the check reports a problem mentioning "{problem}"'))
def caught(case: Case, problem: str) -> None:
    problems = golden.check_record(_rec(case), case.view())
    assert any(problem in p for p in problems), problems


@given("attempts of the same task that failed, retried a failed call, solved it cleanly, and solved it again")
def attempts(case: Case) -> None:
    failed = record(attempt=1, outcome={"passed": False, "reward": 0.0, "checks": {}, "skipped": []})
    retried = record(attempt=2)
    bad_call = {
        "id": "r",
        "type": "function",
        "function": {"name": "get_user", "arguments": '{"user_id": "x"}'},
    }
    retried["messages"] = (
        retried["messages"][:4]
        + [
            {"role": "assistant", "content": None, "tool_calls": [bad_call]},
            {"role": "tool", "tool_call_id": "r", "content": "Error: not found"},
            {"role": "assistant", "content": None, "tool_calls": [{**bad_call, "id": "r2"}]},
            {"role": "tool", "tool_call_id": "r2", "content": "Error: not found"},
        ]
        + retried["messages"][4:]
    )
    case.attempts = [record(attempt=4), retried, failed, record(attempt=3)]  # in no particular order


@when("the golden trajectory is picked")
def pick(case: Case) -> None:
    case.kept = golden.pick(case.attempts)[("telecom", "t1")]


@then("the third attempt is kept")
def third_kept(case: Case) -> None:
    assert case.kept is not None and case.kept["attempt"] == 3


@then("it is refused")
def refused(case: Case) -> None:
    if case.error is None:
        pytest.fail("a test-split record was made")
