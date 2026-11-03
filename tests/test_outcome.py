"""reflexive.outcome: τ²-bench's scoring behind one function (SEL-39).

Needs τ²-bench with its data (`just setup-dev`); skipped otherwise (e.g. in CI). No API calls:
judged checks are switched off with judges=False. The full sweep over every task is
`just tau2-check-gold`; these tests use a few tasks per domain so `just check` stays fast.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tau2")
from tau2.utils.utils import DATA_DIR  # noqa: E402

if not (DATA_DIR / "tau2" / "domains").exists():
    pytest.skip("τ²-bench data folder missing (run `just tau2`)", allow_module_level=True)

from loguru import logger  # noqa: E402

from reflexive.outcome import from_openai, scorer, to_openai  # noqa: E402

logger.disable("tau2")

# (domain, task_id): one with several state changes, plus a telecom task that also checks actions.
CASES = [("retail", "0"), ("airline", "1"), ("telecom", None)]


def _task_id(domain, task_id):
    if task_id is not None:
        return task_id
    sc = scorer(domain)  # a telecom task whose reward also includes ACTION
    return next(t.id for t in sc.tasks.values()
                if "ACTION" in {b.value for b in t.evaluation_criteria.reward_basis})


@pytest.mark.parametrize("domain,task_id", CASES)
def test_correct_actions_pass(domain, task_id):
    sc, tid = scorer(domain), _task_id(domain, task_id)
    out = sc.score(tid, sc.gold_trajectory(tid), "user_stop", judges=False)
    assert out.passed and out.reward == 1.0
    assert all(v == 1.0 for v in out.checks.values())


@pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
def test_missing_state_change_fails(domain):
    sc = scorer(domain)  # first task whose correct answer changes state
    tid = next(t for t in sc.tasks if sc.write_action_indices(t))
    last_write = sc.write_action_indices(tid)[-1]
    out = sc.score(tid, sc.gold_trajectory(tid, drop=last_write), "user_stop", judges=False)
    assert not out.passed and out.reward == 0.0


@pytest.mark.parametrize("termination", ["max_steps", "too_many_errors", "context_window_exceeded"])
def test_premature_stop_scores_zero_even_with_correct_actions(termination):
    sc = scorer("telecom")
    tid = next(iter(sc.tasks))
    out = sc.score(tid, sc.gold_trajectory(tid), termination)
    assert not out.passed and out.reward == 0.0 and "prematurely" in (out.note or "")


def test_both_message_formats_score_the_same():
    sc = scorer("telecom")
    tid = _task_id("telecom", None)
    native = from_openai(sc.gold_trajectory(tid))  # τ²-bench Message objects
    a = sc.score(tid, native, "user_stop")
    b = sc.score(tid, to_openai(native), "user_stop")  # round-tripped through OpenAI dicts
    assert a.reward == b.reward == 1.0 and a.checks == b.checks


def test_judged_checks_are_skipped_and_reported():
    sc = scorer("retail")
    out = sc.score("0", sc.gold_trajectory("0"), "user_stop", judges=False)
    assert out.skipped == ["NL_ASSERTION"] and not out.judged
    assert set(out.checks) == {"DB"}
    assert scorer("telecom").score(_task_id("telecom", None), [], "user_stop").skipped == []


def test_from_openai_shapes():
    msgs = from_openai([
        {"role": "system", "content": "policy"},  # dropped
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a1", "type": "function",
             "function": {"name": "get_user_details", "arguments": json.dumps({"user_id": "u"})}}]},
        {"role": "tool", "tool_call_id": "a1", "content": {"name": "x"}},
        {"role": "user", "content": "done", "tool_calls": [
            {"id": "u1", "name": "toggle_airplane_mode", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "u1", "content": "ok"},
    ])
    assert [m.role for m in msgs] == ["assistant", "tool", "user", "tool"]
    assert msgs[0].tool_calls[0].arguments == {"user_id": "u"}
    assert msgs[0].tool_calls[0].requestor == "assistant"
    assert msgs[1].requestor == "assistant" and msgs[1].content == '{"name": "x"}'
    assert msgs[2].tool_calls[0].requestor == "user" and msgs[3].requestor == "user"


def test_unknown_task_and_role_raise():
    with pytest.raises(KeyError, match="No telecom task"):
        scorer("telecom").task("nope")
    with pytest.raises(ValueError, match="Unknown message role"):
        from_openai([{"role": "narrator", "content": "x"}])
