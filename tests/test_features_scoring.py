"""Steps for features/scoring.feature: a pass is τ²-bench's own verdict; broken runs are apart.

The evaluator scenarios replay each task's correct actions (no model, no API) and are skipped
without τ²-bench; the broken-run scenarios are offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from test_golden import record

from reflexive import golden
from reflexive.runner import is_broken

scenarios("features/scoring.feature")


@dataclass
class Scored:
    """A task's replayed actions and τ²-bench's verdict on them."""

    domain: str
    task_id: str
    drop: int | None = None
    outcome: Any = None  # reflexive.outcome.Outcome


def _scorer(domain: str) -> Any:
    pytest.importorskip("tau2")
    from reflexive.outcome import scorer

    return scorer(domain)


@given(
    parsers.re(r"the correct actions for (?P<domain>retail|airline|telecom) task (?P<task>\d+)"),
    target_fixture="case",
)
def correct_actions(domain: str, task: str) -> Scored:
    _scorer(domain)
    return Scored(domain, task)


@given("the correct actions for the first retail task that changes the database", target_fixture="case")
def first_writing_task() -> Scored:
    sc = _scorer("retail")
    return Scored("retail", next(t for t in sc.tasks if sc.write_action_indices(t)))


@given("the correct actions for the first telecom task", target_fixture="case")
def first_telecom_task() -> Scored:
    sc = _scorer("telecom")
    return Scored("telecom", next(iter(sc.tasks)))


@given("without the last change to the database")
def drop_last_write(case: Scored) -> None:
    case.drop = _scorer(case.domain).write_action_indices(case.task_id)[-1]


def _score(case: Scored, ending: str) -> None:
    sc = _scorer(case.domain)
    case.outcome = sc.score(
        case.task_id, sc.gold_trajectory(case.task_id, drop=case.drop), ending, judges=False
    )


@when("they are scored by τ²-bench's evaluator")
def score_normal(case: Scored) -> None:
    _score(case, "user_stop")


@when(parsers.parse('they are scored as a conversation that ended with "{ending}"'))
def score_ending(case: Scored, ending: str) -> None:
    _score(case, ending)


@then("the conversation passes with a reward of 1")
def passes(case: Scored) -> None:
    assert case.outcome.passed and case.outcome.reward == 1.0


@then("the conversation fails with a reward of 0")
def fails(case: Scored) -> None:
    assert not case.outcome.passed and case.outcome.reward == 0.0


@then("the reason says it ended prematurely")
def premature(case: Scored) -> None:
    assert "prematurely" in (case.outcome.note or "")


# --- Broken runs (offline) ----------------------------------------------------------------------

BROKEN = {
    "crashed with an unexpected error": {"termination": "unexpected_error"},
    "stopped because of an API error": {"error": "RateLimitError: too many requests"},
    "hit an infrastructure error": {"termination": "infrastructure_error"},
}


@given(parsers.parse("a saved conversation that {what}"), target_fixture="rec")
def saved_conversation(what: str) -> dict[str, Any]:
    if what == "ended normally, with the customer saying goodbye":
        return record(termination="user_stop")
    return record(**BROKEN[what])


@then("it counts as a broken run")
def broken(rec: dict[str, Any]) -> None:
    assert is_broken(rec)


@then("it does not count as a broken run")
def not_broken(rec: dict[str, Any]) -> None:
    assert not is_broken(rec)


@then("it is never picked as a golden trajectory")
def never_golden(rec: dict[str, Any]) -> None:
    assert (golden.why_not_clean(rec) or "").startswith("run broke")
    assert golden.pick([rec]) == {}
