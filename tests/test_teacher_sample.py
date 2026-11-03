"""scripts/teacher_sample.py: the fixed task sample, the agents' plumbing and the summary."""

from __future__ import annotations

import importlib.util
import json
import pathlib
from types import SimpleNamespace as NS

import pytest

from reflexive.teacher import tool_errors

_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "teacher_sample.py"


def load():
    spec = importlib.util.spec_from_file_location("teacher_sample", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_wilson_matches_the_table_we_planned_with():
    ts = pytest.importorskip("loguru") and load()
    lo, hi = ts.wilson(0, 30)
    assert lo == 0 and round(hi, 3) == 0.114
    lo, hi = ts.wilson(3, 30)
    assert (round(lo, 3), round(hi, 3)) == (0.035, 0.256)


def test_tool_errors_counts_unknown_tools_and_retried_failures():
    def call(i, name, args):
        return {"role": "assistant", "content": None,
                "tool_calls": [{"id": i, "function": {"name": name, "arguments": json.dumps(args)}}]}

    msgs = [
        call("a", "check_apn_settings", {}),
        {"role": "tool", "tool_call_id": "a", "content": "Error: Tool 'check_apn_settings' not found."},
        call("b", "exchange", {"item_ids": ["1012345678"]}),
        {"role": "tool", "tool_call_id": "b", "content": "Error: Number of 1012345678 not found."},
        call("c", "exchange", {"item_ids": ["1012345678"]}),  # identical retry of a failure
        {"role": "tool", "tool_call_id": "c", "content": "Error: Number of 1012345678 not found."},
        call("d", "exchange", {"item_ids": ["1151293680"]}),  # different args: not a retry
        {"role": "tool", "tool_call_id": "d", "content": '{"status": "exchange requested"}'},
    ]
    assert tool_errors(msgs) == {"unknown_tools": 1, "repeated_failures": 1}


def test_native_agent_adds_no_think_and_reads_tool_calls(monkeypatch):
    ts = load()
    seen = {}

    def fake(**kw):
        seen.update(kw)
        msg = NS(content=None, tool_calls=[NS(id="q1", function=NS(name="get_order_details",
                                                                 arguments='{"order_id": "#W1"}'))])
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=1000, completion_tokens=10))

    import litellm

    monkeypatch.setattr(litellm, "completion", fake)
    agent = ts.NativeAgent("openrouter/qwen/qwen3-32b", ts.PRICES["qwen"], no_think=True)
    turn = agent.turn([{"role": "system", "content": "POLICY"}, {"role": "user", "content": "hi"}],
                      [{"type": "function", "function": {"name": "get_order_details"}}])
    assert seen["messages"][0]["content"].endswith("/no_think") and seen["tools"]
    assert turn["tool_calls"][0]["function"]["name"] == "get_order_details"
    assert agent.usage_dict()["est_cost"] == pytest.approx(1000 * 0.08e-6 + 10 * 0.28e-6, abs=1e-4)


def test_summary_counts_conversations_with_each_problem():
    ts = load()

    def rec(domain, passed, unseen=(), unknown=0, retried=0, tools=3, ended="user_stop"):
        return {"domain": domain, "outcome": {"passed": passed}, "error": None, "user_cost": 0.05,
                "termination": ended,
                "wall_s": 60.0, "agent_usage": {"est_cost": 0.05},
                "diagnostics": {"unseen_arguments": list(unseen), "unknown_tools": unknown,
                                "repeated_failures": retried, "tools_ran": tools}}

    s = ts.summarise("aya", [rec("retail", True), rec("retail", False, unseen=["1012345678"], retried=2),
                             rec("airline", False, unknown=4, tools=0),
                             rec("airline", False, tools=0, ended="unexpected_error")])
    assert (s["passed"], s["n"]) == (1, 4) and s["by_domain"] == {"retail": (1, 2), "airline": (0, 2)}
    assert (s["invented_ids"], s["unknown_tools"], s["repeated_failures"], s["no_tools"]) == (1, 1, 1, 2)
    assert s["errors"] == 1  # the broken run is reported, not silently counted as a model failure
    assert s["cost"] == pytest.approx(0.40)


def test_task_sample_is_fixed_train_only_and_needs_a_state_change():
    pytest.importorskip("tau2")
    from tau2.registry import registry

    from reflexive.outcome import scorer

    ts = load()
    tasks = ts.sample_tasks()
    assert tasks == ts.sample_tasks()  # same every run
    assert sum(d == "retail" for d, _ in tasks) == 20 and sum(d == "airline" for d, _ in tasks) == 10
    for d, t in tasks:
        assert t in registry.get_task_splits_loader(d)()["train"]
        assert scorer(d).write_action_indices(t)


def test_tiny_uses_tiny_aya_prompted_inside_its_8k_window(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-placeholder")
    ts = load()
    agent = ts.make_agent("tiny")()
    assert agent.client.model == "tiny-aya-global" and agent.client.mode == "prompted"
    assert agent.client.style == "cohere" and agent.usage_dict()["est_cost"] == 0.0
    assert ts.CONTEXT["tiny"] == (8192, 1024) and "aya" not in ts.CONTEXT
