"""Steps for features/hard_subset.feature: every train task, grouped by Arm A's result and the teacher's fix.

Offline: uses the made-up Arm A records from test_hard.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from test_hard import CHAT, SRC, arm, gold, train

from reflexive import hard, splits
from reflexive.hallucination import domain_tools

scenarios("features/hard_subset.feature")


@dataclass
class Build:
    """Arm A's records and the golden fixes going in, and what came out."""

    arm: list[dict[str, Any]] = field(default_factory=lambda: [arm("retail", t) for t in train("retail")])
    fixes: list[dict[str, Any]] = field(default_factory=list)
    task: str = ""
    records: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    error: hard.HardSubsetError | None = None
    labels: tuple[list[str], dict[str, Any]] | None = None


@given(parsers.parse("a retail train task that Arm A {result}"), target_fixture="b")
def task_with_result(result: str) -> Build:
    b = Build()
    b.task = train("retail")[0]
    b.arm[0] = arm("retail", b.task, passed=result == "passed")
    return b


@given(parsers.parse("the teacher {fix}"))
def teacher(b: Build, fix: str) -> None:
    b.fixes = [gold("retail", b.task)] if fix == "solved it" else []


@given(parsers.re(r"Arm A's records include a retail (?P<split>dev|test) task"), target_fixture="b")
def extra_task(split: str) -> Build:
    b = Build()
    b.task = splits.task_ids("retail", split)[0]
    b.arm.append(arm("retail", b.task))
    return b


@given("a telecom train task in which Arm A called one of the customer's phone tools", target_fixture="b")
def customer_tool_call() -> Build:
    _, customer = domain_tools("telecom")
    call = {"id": "c1", "type": "function", "function": {"name": customer[0], "arguments": "{}"}}
    view = [*CHAT, {"role": "assistant", "content": "", "tool_calls": [call]}]
    return Build(arm=[arm("telecom", train("telecom")[0], agent_view=view)])


@when("the hard subset is built")
def built(b: Build) -> None:
    try:
        b.records, b.report = hard.build(b.arm, b.fixes, SRC)
    except hard.HardSubsetError as e:
        b.error = e


@when("its failure labels are read")
def read_labels(b: Build) -> None:
    b.labels = hard.labels_of(b.arm[0])


@then(parsers.parse('the task is in the "{group}" group'))
def in_group(b: Build, group: str) -> None:
    assert next(r for r in b.records if r["task_id"] == b.task)["group"] == group


@then('they are "TOOL" and marked as a customer tool')
def tool_and_customer(b: Build) -> None:
    assert b.labels is not None
    labels, detail = b.labels
    assert labels == ["TOOL"] and detail["customer_tool"]


@then("the dev task isn't in it and is counted as left out")
def dev_left_out(b: Build) -> None:
    assert b.task not in {r["task_id"] for r in b.records}
    assert b.report["dev_left_out"] == {"retail": 1}


@then("the build stops, naming the test task")
def stops(b: Build) -> None:
    if b.error is None:
        pytest.fail("the build went ahead with a test task")
    assert any(b.task in line and "TEST" in line for line in b.error.lines)
