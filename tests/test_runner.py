"""reflexive.runner: the agents' plumbing, model names, OpenRouter prices and credits, run checks."""

from __future__ import annotations

import sys
from types import SimpleNamespace as NS

import pytest

from reflexive import runner


def fake_module(monkeypatch, name, **attrs):
    """A stand-in for an optional dependency (litellm, httpx), so CI needs neither installed."""
    import types

    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_native_agent_adds_no_think_and_reads_tool_calls(monkeypatch):
    seen = {}

    def fake(**kw):
        seen.update(kw)
        msg = NS(
            content=None,
            tool_calls=[NS(id="q1", function=NS(name="get_order_details", arguments='{"order_id": "#W1"}'))],
        )
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=1000, completion_tokens=10))

    litellm = fake_module(monkeypatch, "litellm", suppress_debug_info=False, completion=None)

    monkeypatch.setattr(litellm, "completion", fake)
    agent = runner.NativeAgent("openrouter/qwen/qwen3-32b", runner.PRICES["qwen"], no_think=True)
    turn = agent.turn(
        [{"role": "system", "content": "POLICY"}, {"role": "user", "content": "hi"}],
        [{"type": "function", "function": {"name": "get_order_details"}}],
    )
    assert seen["messages"][0]["content"].endswith("/no_think") and seen["tools"]
    assert turn["tool_calls"][0]["function"]["name"] == "get_order_details"
    assert agent.usage_dict()["est_cost"] == pytest.approx(1000 * 0.08e-6 + 10 * 0.28e-6, abs=1e-4)


def test_tiny_uses_tiny_aya_prompted_inside_its_8k_window(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-placeholder")
    agent = runner.make_agent("tiny")()
    assert isinstance(agent, runner.AyaAgent)
    assert agent.client.model == "tiny-aya-global" and agent.client.mode == "prompted"
    assert agent.client.style == "cohere" and agent.usage_dict()["est_cost"] == 0.0
    assert runner.CONTEXT == {"tiny": (8192, 1024)}  # only Tiny Aya is cut to 8K


def test_qwen_retries_an_empty_reply(monkeypatch):
    replies = iter([NS(content="", tool_calls=None), NS(content="How can I help?", tool_calls=None)])

    def fake(**kw):
        return NS(choices=[NS(message=next(replies))], usage=NS(prompt_tokens=10, completion_tokens=1))

    litellm = fake_module(monkeypatch, "litellm", suppress_debug_info=False, completion=None)

    monkeypatch.setattr(litellm, "completion", fake)
    agent = runner.NativeAgent("openrouter/qwen/qwen3-32b", runner.PRICES["qwen"], no_think=True)
    assert agent.turn([{"role": "user", "content": "hi"}], []) == {
        "role": "assistant",
        "content": "How can I help?",
    }
    assert agent.usage["calls"] == 2 and agent.usage["empty"] == 1


def test_tiny_counts_tokens_more_cautiously():
    msgs = [{"role": "user", "content": "x" * 4000}]
    assert runner.COUNT["tiny"](msgs) > runner.estimate_tokens(msgs)


FAKE_LIST = {
    "z-ai/glm-4.7-flash": {
        "id": "z-ai/glm-4.7-flash",
        "hugging_face_id": "zai-org/GLM-4.7-Flash",
        "pricing": {"prompt": "0.00000007", "completion": "0.0000004"},
    },
    "anthropic/claude-haiku-5.5": {
        "id": "anthropic/claude-haiku-5.5",
        "hugging_face_id": "",
        "supported_parameters": ["tools", "tool_choice", "max_tokens"],
        "pricing": {"prompt": "0.000001", "completion": "0.000005"},
    },
}


def test_bakeoff_slate_prices_and_open_weights_come_from_openrouter(monkeypatch):
    monkeypatch.setattr(runner, "_openrouter_models", lambda: FAKE_LIST)
    glm = runner.make_agent("glm")()
    assert isinstance(glm, runner.OpenRouterAgent)
    assert glm.slug == "z-ai/glm-4.7-flash" and glm.prices == (7e-8, 4e-7)
    assert runner.open_weights("glm") is True and runner.open_weights("haiku") is False
    haiku = runner.make_agent("or:anthropic/claude-haiku-5.5")()
    assert isinstance(haiku, runner.OpenRouterAgent) and haiku.supported is not None
    assert haiku.prices == (1e-6, 5e-6) and "temperature" not in haiku.supported
    assert runner.resolve_models(["aya", "bakeoff", "glm"])[:3] == ["aya", "glm", "glm53f"]
    assert runner.file_name("or:moonshotai/kimi-k2.6") == "or-moonshotai__kimi-k2.6"


def test_removed_models_stop_the_run():
    assert runner.model_gone({"error": "NotFoundError: model 'x' was removed on April 4, 2026"})
    assert not runner.model_gone({"error": "RateLimitError"}) and not runner.model_gone({})


def test_temperature_is_only_sent_when_the_model_accepts_it(monkeypatch):
    import types

    sent = []
    msg = types.SimpleNamespace(tool_calls=None, content="Hello")
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)
    fake = types.SimpleNamespace(suppress_debug_info=False, completion=lambda **kw: sent.append(kw) or resp)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    runner.NativeAgent("openrouter/a/b", (0, 0), supported={"tools"}).turn([], [])
    runner.NativeAgent("openrouter/a/c", (0, 0), supported={"tools", "temperature"}).turn([], [])
    assert "temperature" not in sent[0] and sent[1]["temperature"] == 0.0
    assert sent[0]["extra_body"] == {"provider": {"require_parameters": True}}


def test_openrouter_agent_hands_reasoning_back_and_retries_provider_errors(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-placeholder")
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    replies = [
        {"choices": [{"finish_reason": "error", "message": {}}]},  # Gemini-style failure: retried
        {
            "provider": "Google",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "reasoning_details": [{"type": "reasoning.encrypted", "data": "sig"}],
                        "tool_calls": [{"id": "c1", "function": {"name": "find_user", "arguments": "{}"}}],
                    },
                }
            ],
        },
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
    monkeypatch.setattr(
        httpx, "post", lambda url, json, headers, timeout: bodies.append(json) or R(replies.pop(0))
    )
    agent = runner.OpenRouterAgent("google/x", (0.0, 0.0), supported={"tools"})
    first = agent.turn([{"role": "user", "content": "hi"}], [])
    assert first["tool_calls"][0]["id"] == "c1" and "temperature" not in bodies[0]
    agent.turn(
        [{"role": "user", "content": "hi"}, first, {"role": "tool", "tool_call_id": "c1", "content": "{}"}],
        [],
    )
    assert bodies[-1]["messages"][1]["reasoning_details"][0]["data"] == "sig"
    assert agent.usage_dict()["est_cost"] == 0.001 and agent.usage["providers"] == {"Google": 2}


def test_out_of_credits_is_recognised():
    assert runner.out_of_credits({"error": "OutOfCredits: OpenRouter 402: ..."})
    assert runner.out_of_credits({"error": "HTTPStatusError: Client error '402 Payment Required'"})
    assert not runner.out_of_credits({"error": "RateLimitError"}) and not runner.out_of_credits({})


def test_qwen_thinking_off_variant_sends_reasoning_disabled(monkeypatch):
    monkeypatch.setattr(runner, "_openrouter_models", lambda: {})
    agent = runner.make_agent("qwen38nt")()
    assert isinstance(agent, runner.OpenRouterAgent)
    assert agent.slug == "qwen/qwen3.8-27b" and agent.extra_body == {"reasoning": {"enabled": False}}
    assert "qwen38nt" not in runner.resolve_models(["bakeoff"])


def test_in_flight_budget_is_not_out_of_credits():
    busy = 'OutOfCredits: OpenRouter 402: {"reason":"in_flight_budget_exhausted"}'
    assert not runner.out_of_credits({"error": busy})
    assert runner.out_of_credits({"error": "OutOfCredits: OpenRouter 402: insufficient credits"})


def test_the_dropped_models_are_gone_and_an_unknown_name_says_what_exists() -> None:
    """aya8, aya-steps and gpt never produced a results file; tiny and qwen did (phase 1)."""
    for name in ("aya8", "aya-steps", "gpt"):
        assert name not in runner.MODELS
        with pytest.raises(ValueError, match="choose from"):
            runner.make_agent(name)
    assert {"aya", "tiny", "qwen", "glm53f", "qwen38nt"} <= set(runner.MODELS)


def test_variants_count_as_openrouter_models() -> None:
    """So their run checks OpenRouter credits first (the bake-off script used to skip it)."""
    assert runner.uses_openrouter("qwen38nt") and runner.uses_openrouter("or:a/b")
    assert not runner.uses_openrouter("aya") and not runner.uses_openrouter("tiny")
