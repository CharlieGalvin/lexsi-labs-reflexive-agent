"""Steps for features/splits.feature: the frozen τ²-bench splits (SEL-36)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pytest_bdd import given, scenarios, then, when

from reflexive import splits

scenarios("features/splits.feature")


@dataclass
class Check:
    """A task, and what happened when code asked whether it may be trained on."""

    domain: str
    task_id: str
    error: Exception | None = None


@given("the frozen splits", target_fixture="frozen")
def frozen() -> dict[str, dict[str, set[str]]]:
    return {d: {s: set(splits.task_ids(d, s)) for s in splits.SPLITS} for d in splits.DOMAINS}


@then("there are 142 train, 36 dev and 100 test tasks")
def totals(frozen: dict[str, dict[str, set[str]]]) -> None:
    counts = {s: sum(len(frozen[d][s]) for d in splits.DOMAINS) for s in splits.SPLITS}
    assert counts == {"train": 142, "dev": 36, "test": 100}


@then("the dev split has 15 retail, 6 airline and 15 telecom tasks")
def dev_sizes(frozen: dict[str, dict[str, set[str]]]) -> None:
    assert {d: len(frozen[d]["dev"]) for d in splits.DOMAINS} == {"retail": 15, "airline": 6, "telecom": 15}


@then("no task appears in more than one split, in any domain")
def disjoint(frozen: dict[str, dict[str, set[str]]]) -> None:
    for d, by_split in frozen.items():
        train, dev, test = (by_split[s] for s in splits.SPLITS)
        assert not (train & dev or train & test or dev & test), d


@given("a retail task from the test split", target_fixture="check")
def a_test_task() -> Check:
    return Check("retail", splits.task_ids("retail", "test")[0])


@given("a retail task from the train split", target_fixture="check")
def a_train_task() -> Check:
    return Check("retail", splits.task_ids("retail", "train")[0])


@when("code checks whether the task may be trained on")
def may_train(check: Check) -> None:
    try:
        splits.assert_not_test(check.domain, check.task_id)
    except ValueError as e:
        check.error = e


@then("it is refused with an error saying it is a TEST task")
def refused(check: Check) -> None:
    assert check.error is not None and "TEST" in str(check.error)


@then("it is allowed")
def allowed(check: Check) -> None:
    if check.error is not None:
        pytest.fail(f"train task refused: {check.error}")
