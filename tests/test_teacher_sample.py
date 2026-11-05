"""scripts/teacher/bakeoff.py: the fixed task sample, the agents' plumbing and the summary."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace as NS

import pytest
from _scripts import load_script

from reflexive.teacher import tool_errors


def fake_module(monkeypatch, name, **attrs):
    """A stand-in for an optional dependency (litellm, httpx), so CI needs neither installed."""
    import types

    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def load():
    return load_script("teacher/bakeoff")


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

    litellm = fake_module(monkeypatch, "litellm", suppress_debug_info=False, completion=None)

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


def test_overlap_lists_tasks_only_the_teacher_solves(capsys):
    ts = load()
    ts.print_overlap({"qwen": {("retail", "4"), ("airline", "7")},
                      "aya8": {("airline", "7"), ("retail", "9")}})
    out = capsys.readouterr().out
    assert "qwen → aya8   1   (1 the other way, 1 both)  retail:4" in out


def test_aya_steps_is_the_teacher_with_hints_and_qwen_retries_an_empty_reply(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-placeholder")
    ts = load()
    assert ts.HINTS == {"aya-steps": "steps"} and ts.make_agent("aya-steps")().client.model == \
        "c4ai-aya-expanse-32b"
    replies = iter([NS(content="", tool_calls=None), NS(content="How can I help?", tool_calls=None)])

    def fake(**kw):
        return NS(choices=[NS(message=next(replies))], usage=NS(prompt_tokens=10, completion_tokens=1))

    litellm = fake_module(monkeypatch, "litellm", suppress_debug_info=False, completion=None)

    monkeypatch.setattr(litellm, "completion", fake)
    agent = ts.NativeAgent("openrouter/qwen/qwen3-32b", ts.PRICES["qwen"], no_think=True)
    assert agent.turn([{"role": "user", "content": "hi"}], []) == {
        "role": "assistant", "content": "How can I help?"}
    assert agent.usage["calls"] == 2 and agent.usage["empty"] == 1


def test_tiny_counts_tokens_more_cautiously():
    ts = load()
    msgs = [{"role": "user", "content": "x" * 4000}]
    assert ts.COUNT["tiny"](msgs) > ts.estimate_tokens(msgs)


def test_aya8_is_self_hosted_inside_its_8k_window(monkeypatch):
    ts = load()
    monkeypatch.delenv("AYA8_API_BASE", raising=False)
    with pytest.raises(ts.ModelUnavailable, match="removed 4 Apr 2026"):
        ts.make_agent("aya8")()
    monkeypatch.setenv("AYA8_API_BASE", "http://localhost:11434/v1")
    monkeypatch.setenv("AYA8_MODEL", "aya-expanse:8b")
    c = ts.make_agent("aya8")().client
    assert (c.model, c.api_base) == ("aya-expanse:8b", "http://localhost:11434/v1")
    assert ts.CONTEXT["aya8"] == (8192, 1024)


FAKE_LIST = {
    "z-ai/glm-4.7-flash": {"id": "z-ai/glm-4.7-flash", "hugging_face_id": "zai-org/GLM-4.7-Flash",
                           "pricing": {"prompt": "0.00000007", "completion": "0.0000004"}},
    "anthropic/claude-haiku-5.5": {"id": "anthropic/claude-haiku-5.5", "hugging_face_id": "",
                                   "supported_parameters": ["tools", "tool_choice", "max_tokens"],
                                   "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
}


def test_bakeoff_slate_prices_and_open_weights_come_from_openrouter(monkeypatch):
    ts = load()
    monkeypatch.setattr(ts, "_openrouter_models", lambda: FAKE_LIST)
    glm = ts.make_agent("glm")()
    assert glm.slug == "z-ai/glm-4.7-flash" and glm.prices == (7e-8, 4e-7)
    assert ts._open_weights("glm") is True and ts._open_weights("haiku") is False
    haiku = ts.make_agent("or:anthropic/claude-haiku-5.5")()
    assert haiku.prices == (1e-6, 5e-6) and "temperature" not in haiku.supported
    assert ts.resolve_models(["aya", "bakeoff", "glm"])[:3] == ["aya", "glm", "glm53f"]
    assert ts._file_name("or:moonshotai/kimi-k2.6") == "or-moonshotai__kimi-k2.6"


def test_removed_models_stop_the_run():
    ts = load()
    assert ts._gone({"error": "NotFoundError: model 'x' was removed on April 4, 2026"})
    assert not ts._gone({"error": "RateLimitError"}) and not ts._gone({})


def test_temperature_is_only_sent_when_the_model_accepts_it(monkeypatch):
    import types

    ts = load()
    sent = []
    msg = types.SimpleNamespace(tool_calls=None, content="Hello")
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)
    fake = types.SimpleNamespace(suppress_debug_info=False,
                                 completion=lambda **kw: sent.append(kw) or resp)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    ts.NativeAgent("openrouter/a/b", (0, 0), supported={"tools"}).turn([], [])
    ts.NativeAgent("openrouter/a/c", (0, 0), supported={"tools", "temperature"}).turn([], [])
    assert "temperature" not in sent[0] and sent[1]["temperature"] == 0.0
    assert sent[0]["extra_body"] == {"provider": {"require_parameters": True}}


def test_openrouter_agent_hands_reasoning_back_and_retries_provider_errors(monkeypatch):
    ts = load()
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    replies = [
        {"choices": [{"finish_reason": "error", "message": {}}]},  # Gemini-style failure: retried
        {"provider": "Google", "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
         "choices": [{"finish_reason": "tool_calls", "message": {
             "content": None, "reasoning_details": [{"type": "reasoning.encrypted", "data": "sig"}],
             "tool_calls": [{"id": "c1", "function": {"name": "find_user", "arguments": "{}"}}]}}]},
        {"provider": "Google", "usage": {}, "choices": [{"message": {"content": "Done."}}]},
    ]
    bodies = []

    class R:
        def __init__(self, data):
            self.data, self.status_code, self.text = data, 200, ""

        def json(self):
            return self.data

        def raise_for_status(self):
            pass

    httpx = fake_module(monkeypatch, "httpx", HTTPError=Exception, post=None)
    monkeypatch.setattr(httpx, "post",
                        lambda url, json, headers, timeout: bodies.append(json) or R(replies.pop(0)))
    agent = ts.OpenRouterAgent("google/x", (0.0, 0.0), supported={"tools"})
    first = agent.turn([{"role": "user", "content": "hi"}], [])
    assert first["tool_calls"][0]["id"] == "c1" and "temperature" not in bodies[0]
    agent.turn([{"role": "user", "content": "hi"}, first,
                {"role": "tool", "tool_call_id": "c1", "content": "{}"}], [])
    assert bodies[-1]["messages"][1]["reasoning_details"][0]["data"] == "sig"
    assert agent.usage_dict()["est_cost"] == 0.001 and agent.usage["providers"] == {"Google": 2}


def test_out_of_credits_is_recognised():
    ts = load()
    assert ts._no_credits({"error": "OutOfCredits: OpenRouter 402: ..."})
    assert ts._no_credits({"error": "HTTPStatusError: Client error '402 Payment Required'"})
    assert not ts._no_credits({"error": "RateLimitError"}) and not ts._no_credits({})


def test_qwen_thinking_off_variant_sends_reasoning_disabled(monkeypatch):
    ts = load()
    monkeypatch.setattr(ts, "_openrouter_models", lambda: {})
    agent = ts.make_agent("qwen38nt")()
    assert agent.slug == "qwen/qwen3.8-27b" and agent.extra_body == {"reasoning": {"enabled": False}}
    assert "qwen38nt" not in ts.resolve_models(["bakeoff"])


def test_in_flight_budget_is_not_out_of_credits():
    ts = load()
    busy = 'OutOfCredits: OpenRouter 402: {"reason":"in_flight_budget_exhausted"}'
    assert not ts._no_credits({"error": busy})
    assert ts._no_credits({"error": "OutOfCredits: OpenRouter 402: insufficient credits"})
