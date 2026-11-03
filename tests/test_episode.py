"""reflexive.episode: a τ²-bench conversation our code drives (SEL-32).

No API calls: the simulated customer is LiteLLM's `mock_response` (a fixed reply), and the
agent turns are each task's correct actions. Needs τ²-bench with its data; skipped otherwise.
"""

from __future__ import annotations

import inspect
import json

import pytest

pytest.importorskip("tau2")
from tau2.utils.utils import DATA_DIR  # noqa: E402

if not (DATA_DIR / "tau2" / "domains").exists():
    pytest.skip("τ²-bench data folder missing (run `just tau2`)", allow_module_level=True)

from loguru import logger  # noqa: E402

from reflexive.episode import TRUNCATED, Tau2Episode, compact_json, fit_context  # noqa: E402
from reflexive.outcome import scorer  # noqa: E402

logger.disable("tau2")
CUSTOMER = {"user_llm": "gpt-4.1", "user_llm_args": {"mock_response": "Yes, please go ahead."}}


def play_correct_actions(domain, task_id, **kw):
    ep = Tau2Episode(domain, task_id, **CUSTOMER, **kw)
    opener = ep.reset()
    seen = []
    for m in scorer(domain).gold_trajectory(task_id):
        if m["role"] == "assistant" and not ep.done:
            ep.prompt()
            if ep.done:
                break
            seen.append(ep.step({"role": "assistant", "content": None,
                                 "tool_calls": m["tool_calls"]}))
    if not ep.done:
        ep.step({"role": "assistant", "content": "That's all done for you."})
        ep.finish()
    return ep, opener, seen


@pytest.mark.parametrize("domain,task_id", [("retail", "0"), ("airline", "1")])
def test_correct_actions_through_the_episode_pass(domain, task_id):
    ep, opener, seen = play_correct_actions(domain, task_id, max_context=200_000)
    assert [m["role"] for m in opener] == ["assistant", "user"]  # τ²-bench greeting, customer
    assert all(len(new) == 1 and new[0]["role"] == "tool" for new in seen)  # no echo
    rec = ep.record(judges=False)
    assert rec["termination"] == "agent_stop"
    assert rec["outcome"]["passed"] and rec["outcome"]["reward"] == 1.0
    json.dumps(rec)  # saveable as a JSON line


@pytest.mark.parametrize("domain", ["retail", "airline", "telecom"])
def test_prompt_and_tools_match_the_official_agent(domain):
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import build_text_orchestrator

    sc = scorer(domain)
    task_id = next(iter(sc.tasks))
    ep = Tau2Episode(domain, task_id, **CUSTOMER)
    ep.reset()
    official = build_text_orchestrator(
        TextRunConfig(domain=domain, llm_agent="gpt-4.1", llm_user="gpt-4.1"), sc.task(task_id)
    ).agent
    assert ep.system_prompt == official.system_prompt
    assert [t["function"]["name"] for t in ep.tool_schemas] == [t.name for t in official.tools]
    assert "done" not in {t["function"]["name"] for t in ep.tool_schemas}
    ep.finish()


def test_tool_results_are_compact_json():
    ep, _, seen = play_correct_actions("retail", "0", max_context=200_000)
    results = [new[0]["content"] for new in seen if new[0]["content"].startswith("{")]
    assert results and all("\n" not in r and ": " not in r for r in results)
    assert compact_json('{\n  "a": 1\n}') == '{"a":1}' and compact_json("not json") == "not json"


def test_context_guard_trims_oldest_tool_results_first():
    msgs = [{"role": "system", "content": "s"}] + [
        {"role": "tool", "tool_call_id": str(i), "content": "x" * 400} for i in range(4)]
    out, trimmed, fits = fit_context(msgs, budget=250)
    assert fits and trimmed >= 1
    assert out[1]["content"] == TRUNCATED and out[-1]["content"] == "x" * 400  # latest kept
    assert msgs[1]["content"] == "x" * 400  # input untouched
    _, _, fits = fit_context(msgs, budget=10)
    assert not fits


def test_conversation_that_cannot_fit_ends_as_context_overflow():
    ep, _, _ = play_correct_actions("retail", "0", max_context=6000, reserve=500)
    rec = ep.record(judges=False)
    assert rec["termination"] == "context_window_exceeded"
    assert rec["flags"]["context_overflow"] and not rec["outcome"]["passed"]
    assert rec["outcome"]["reward"] == 0.0


def test_trimming_lets_a_tight_conversation_finish_and_keeps_the_full_trajectory():
    # retail 0 peaks at ~6,500 estimated tokens; a 6,200 window forces trimming but still fits
    ep, _, _ = play_correct_actions("retail", "0", max_context=6200, reserve=200)
    rec = ep.record(judges=False)
    assert rec["flags"]["tool_results_trimmed"] > 0 and not rec["flags"]["context_overflow"]
    assert rec["termination"] == "agent_stop" and rec["outcome"]["passed"]
    assert TRUNCATED not in json.dumps(rec["messages"])  # only the model's view was trimmed
    assert rec["flags"]["peak_prompt_tokens"] > 6200


def test_tool_functions_match_the_tools_and_run_in_the_conversation():
    ep = Tau2Episode("retail", "0", **CUSTOMER)
    ep.reset()
    fns = {f.__name__: f for f in ep.tool_functions()}
    assert set(fns) == {t["function"]["name"] for t in ep.tool_schemas}
    find = fns["find_user_id_by_name_zip"]
    assert list(inspect.signature(find).parameters) == ["first_name", "last_name", "zip"]
    assert find.__doc__
    first = scorer("retail").task("0").evaluation_criteria.actions[0]
    assert find(**first.arguments) == "yusuf_rossi_9620"
    ep.finish()
    calls = [tc["function"]["name"] for m in ep.trajectory() for tc in m.get("tool_calls") or []]
    assert "find_user_id_by_name_zip" in calls


def test_unknown_task_fails_fast():
    with pytest.raises(KeyError):
        Tau2Episode("retail", "no-such-task", **CUSTOMER)


def test_empty_agent_turn_is_refused_and_the_conversation_carries_on():
    ep = Tau2Episode("retail", "0", **CUSTOMER)
    ep.reset()
    with pytest.raises(ValueError, match="message or at least one tool call"):
        ep.step({"role": "assistant", "content": "   "})
    assert not ep.done
    new = ep.step({"role": "assistant", "content": "How can I help with your order?"})
    assert new and new[0]["role"] == "user"
    ep.finish()
    assert ep.record(judges=False)["termination"] == "agent_stop"
