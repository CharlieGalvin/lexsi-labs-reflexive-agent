"""Is Aya Expanse 32B a usable teacher? A fixed 30-task test against a reference model (SEL-37).

    python scripts/teacher_sample.py                      # Aya vs Qwen3 32B (same size)
    python scripts/teacher_sample.py --models aya qwen gpt
    python scripts/teacher_sample.py --models aya --workers 2

**Tasks.** A fixed sample of 30 τ²-bench *train* tasks whose correct answer changes the database:
20 retail + 10 airline (seed 0, the same every run). On these, doing nothing can't pass.

**Scoring.** τ²-bench's database / state checks only (no LLM judges, so no OpenAI key needed).
That is a *best case*: failing it is a definite fail, but passing it doesn't guarantee the
judged checks would pass too. If even the best case is low, the model is a weak teacher.

**Models** (all as the agent through `Tau2Episode`, GPT-4.1 as the simulated customer):
- ``aya``: Aya Expanse 32B via Cohere, prompted tool calls (Aya's own format); our teacher.
- ``qwen``: Qwen3 32B via OpenRouter, native tool calling, thinking off (``/no_think``): a
  same-size model trained for tool use, so it separates "Aya is weak" from "32B is too small".
- ``gpt``: GPT-4.1 via OpenRouter (or OpenAI), native tool calling: shows the tasks are solvable.
- ``glm``: GLM-4.7-Flash (``z-ai/glm-4.7-flash``) via OpenRouter, native tool calling, thinking
  left on: the candidate *teacher*.
- The **teacher bake-off** slate (``--models bakeoff``, see ``SLATE``): open-weight and closed
  API models from the τ²-bench airline leaderboard, all via OpenRouter with native tools. Any
  other OpenRouter model works as ``or:<slug>``, e.g. ``or:moonshotai/kimi-k2.6``. Prices and
  whether the weights are open come live from OpenRouter's public model list.
- ``aya-steps``: the teacher again, with τ²-bench's resolution steps (tool names in order, no
  IDs) added to its prompt, the hinted test from the decision rule. Train tasks only.
- ``aya8``: Aya Expanse 8B, a candidate student, prompted like the others inside its real 8K
  window. Cohere removed it from its API on 4 Apr 2026, so it runs on a server you start
  yourself (Ollama, vLLM...): set ``AYA8_API_BASE`` (an OpenAI-compatible ``/v1`` URL) and
  ``AYA8_MODEL`` (the server's own name for it, e.g. ``aya-expanse:8b`` on Ollama).
- ``tiny``: Tiny Aya (``tiny-aya-global``) via Cohere, our *student*, prompted exactly like the
  teacher, inside its real 8,192-token window (our context guard on; overflows are counted). Its
  API price isn't published, so its cost line is the simulated customer only.

**Per conversation** it also counts tool hallucinations: ID-like arguments never seen earlier
(invented IDs), calls to tools the agent doesn't have, and identical retries of a failed call.

Records go to `results/episodes/sample-<model>.jsonl`; an interrupted run resumes where it
stopped (finished tasks are skipped). Prints pass rates with 95% confidence ranges (Wilson).
"""

from __future__ import annotations

import argparse
import functools
import json
import pathlib
import random
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import sqrt
from typing import Any

try:  # loguru comes with τ²-bench; CI installs the project without it
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="ERROR")
except ImportError:  # pragma: no cover - only without τ²-bench
    logger = None

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from reflexive.episode import Tau2Episode, estimate_tokens  # noqa: E402
from reflexive.teacher import (  # noqa: E402
    ContextTooLong,
    EmptyTeacherReply,
    TeacherClient,
    tool_errors,
    unseen_arguments,
)

SAMPLE = {"retail": 20, "airline": 10}
SEED = 0
# USD per token, for models whose cost LiteLLM doesn't report (listed prices, 3 Nov 2026)
PRICES = {"qwen": (0.08e-6, 0.28e-6), "gpt": (2.0e-6, 8.0e-6), "glm": (0.07e-6, 0.40e-6)}

# Teacher bake-off (SEL-58): OpenRouter slugs. Picked from OpenRouter's τ²-bench airline board
# (3 Nov 2026) to span open vs closed weights and cheap vs strong; scores are that board's.
SLATE = {
    # open weights (expected; the run checks OpenRouter's Hugging Face link)
    "glm": "z-ai/glm-4.7-flash",              # 66.7%, our first pick
    "glm53f": "z-ai/glm-5.3-flash",           # 75.6%
    "dsv4f": "deepseek/deepseek-v4-flash",    # 73.9%
    "mimo": "xiaomi/mimo-v2.6-flash",         # 77.9%
    "qwen38": "qwen/qwen3.8-27b",             # 77.5%, dense 27B
    "gemma4": "google/gemma-4-31b-it",        # 75.3%, dense 31B
    "gptoss": "openai/gpt-oss-120b",          # 65.2%
    # closed weights
    "haiku": "anthropic/claude-haiku-5.5",    # 72.8%
    # gflite (google/gemini-3.5-flash-lite) dropped 3 Nov at Charlie's request; or:<slug> still runs it
    "sonnet": "anthropic/claude-sonnet-5.5",  # 78.8%
    "gpt51": "openai/gpt-5.1",                # 80.0%, the ceiling
    "gem38f": "google/gemini-3.8-flash",      # 71.3%, added 3 Nov at Charlie's request
    "opus": "anthropic/claude-opus-5.5",      # 75.3%, added 3 Nov; the priciest (~$0.34/task)
}

# Request-setting variants of slate models (not part of --models bakeoff)
VARIANTS = {
    "qwen38nt": ("qwen/qwen3.8-27b", {"reasoning": {"enabled": False}}),  # Qwen3.8 27B, thinking off
}


# --- Task sample --------------------------------------------------------------------------------


def sample_tasks(sizes: dict[str, int] = SAMPLE, seed: int = SEED) -> list[tuple[str, str]]:
    """The fixed sample: train tasks whose expected actions change state, per domain."""
    from tau2.registry import registry

    from reflexive.outcome import scorer

    out = []
    for domain, n in sizes.items():
        sc = scorer(domain)
        pool = sorted(t for t in registry.get_task_splits_loader(domain)()["train"]
                      if sc.write_action_indices(t))
        out += [(domain, t) for t in random.Random(seed).sample(pool, min(n, len(pool)))]
    return out


# --- Agents -------------------------------------------------------------------------------------


class NativeAgent:
    """An API model with native tool calling, as the agent (one call per turn)."""

    def __init__(self, model: str, prices: tuple[float, float], no_think: bool = False,
                 supported: set[str] | None = None):
        self.model, self.prices, self.no_think = model, prices, no_think
        # OpenRouter's list of parameters the model accepts. Some newer models (reasoning ones)
        # refuse `temperature`; with require_parameters on, sending it finds no endpoint at all.
        self.supported = supported
        self.temperature = 0.0  # fix runs raise it for retries (scripts/teacher_fixes.py)
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def turn(self, messages: list[dict], tools: list[dict]) -> dict:
        import litellm

        litellm.suppress_debug_info = True
        if self.no_think and messages and messages[0]["role"] == "system":
            messages = [{**messages[0], "content": messages[0]["content"] + "\n\n/no_think"},
                        *messages[1:]]
        for _attempt in range(2):  # Qwen3 sometimes returns an empty reply: ask once more
            start = time.monotonic()
            extra = ({"extra_body": {"provider": {"require_parameters": True}}}  # tools must work
                     if self.model.startswith("openrouter/") else {})
            if not self.supported or "temperature" in self.supported:
                extra["temperature"] = self.temperature
            resp = litellm.completion(model=self.model, messages=messages, tools=tools,
                                      num_retries=3, **extra)
            self.usage["seconds"] += time.monotonic() - start
            self.usage["calls"] += 1
            u = getattr(resp, "usage", None)
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            if msg.tool_calls or (msg.content or "").replace("<think>", "").replace("</think>", "").strip():
                break
            self.usage["empty"] = self.usage.get("empty", 0) + 1
            messages = [*messages, {"role": "user", "content":
                                    "(Write your next turn now: a tool call, or a message to the customer.)"}]
        calls = [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                 for tc in (msg.tool_calls or [])]
        if calls:  # τ²-bench's rule: a tool call or a message, not both
            return {"role": "assistant", "content": None, "tool_calls": calls}
        text = (msg.content or "").replace("<think>", "").replace("</think>", "").strip()
        if not text:
            raise EmptyTeacherReply(f"{self.model} returned an empty reply")
        return {"role": "assistant", "content": text}

    def usage_dict(self) -> dict:
        pin, pout = self.prices
        return {**self.usage, "seconds": round(self.usage["seconds"], 2),
                "est_cost": round(self.usage["prompt_tokens"] * pin
                                  + self.usage["completion_tokens"] * pout, 4)}


class ModelError(RuntimeError):
    """The provider failed mid-reply (finish_reason 'error'), even after retries."""


class OutOfCredits(RuntimeError):
    """OpenRouter refused for lack of credits (HTTP 402): stop the whole run."""


def openrouter_credits() -> float | None:
    """Credits left on the OpenRouter account in USD, or None if it can't be checked."""
    import os

    import httpx

    try:
        r = httpx.get("https://openrouter.ai/api/v1/credits", timeout=20,
                      headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
        d = r.json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:  # noqa: BLE001 - no key, offline, or the endpoint changed: don't block the run
        return None


MIN_CREDITS = 2.0  # USD; below this a model's 30 conversations can run dry halfway


class OpenRouterAgent(NativeAgent):
    """A bake-off model, called on OpenRouter's chat API directly rather than through LiteLLM.

    Why: some newer models (Gemini 3.x, models that think between tool calls) need their
    reasoning handed back on the next turn. OpenRouter returns it as ``reasoning_details`` on
    the reply; τ²-bench rebuilds the history without it, so the next call fails with
    ``finish_reason: "error"``. This agent remembers each reply's ``reasoning_details`` and puts
    them back on the matching assistant message. It also records OpenRouter's billed cost
    and which provider served each call.
    """

    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, slug: str, prices: tuple[float, float], supported: set[str] | None = None):
        super().__init__(f"openrouter/{slug}", prices, supported=supported)
        self.slug = slug
        self.reasoning: dict[str, Any] = {}  # first tool-call id or reply text -> reasoning_details
        self.provider: str | None = None
        self.extra_body: dict[str, Any] = {}  # added to every request (VARIANTS)
        self.usage.update(billed_cost=0.0, providers={})

    def _with_reasoning(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m.get("role") == "assistant":
                key = (m.get("tool_calls") or [{}])[0].get("id") or m.get("content")
                if key in self.reasoning:
                    m = {**m, "reasoning_details": self.reasoning[key]}
            out.append(m)
        return out

    def _post(self, body: dict) -> dict:
        import os

        import httpx

        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        last, waits = None, 0
        for attempt in range(4 + 6):
            if attempt and waits == 0:
                time.sleep(2 ** min(attempt, 4))
            try:
                r = httpx.post(self.URL, json=body, headers=headers, timeout=300)
            except httpx.HTTPError as e:  # network blip or timeout: retry
                last = f"{type(e).__name__}: {e}"
                continue
            if r.status_code == 402 and "in_flight" in r.text and waits < 6:
                # Not empty: too much spend reserved by requests already running. Wait and retry.
                waits += 1
                time.sleep(20)
                last = "OpenRouter in-flight budget"
                continue
            waits = 0
            if r.status_code == 402:  # out of credits: nothing will work until topped up
                raise OutOfCredits(f"OpenRouter 402: {r.text[:200]}")
            if r.status_code == 404 or (r.status_code == 400 and "No endpoints" in r.text):
                raise ModelUnavailable(f"NotFoundError: {r.text[:300]}")
            if r.status_code in (408, 429) or r.status_code >= 500:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                continue
            r.raise_for_status()
            data = r.json()
            choice = (data.get("choices") or [{}])[0]
            if "error" in data or choice.get("finish_reason") == "error":  # failed mid-reply
                last = f"provider error: {json.dumps(data.get('error') or choice.get('error'))[:300]}"
                continue
            return data
        raise ModelError(f"{self.slug}: {last}")

    def turn(self, messages: list[dict], tools: list[dict]) -> dict:
        body: dict[str, Any] = {"model": self.slug, "tools": tools,
                                "provider": {"require_parameters": True},
                                "usage": {"include": True},
                                # a cap keeps OpenRouter's per-request reservation small (it holds
                                # max_tokens × price while a request runs); turns are far shorter
                                "max_tokens": 8192}
        if not self.supported or "temperature" in self.supported:
            body["temperature"] = self.temperature
        if self.provider:  # pin one provider (and its quantisation) so the run is reproducible
            body["provider"] = {"require_parameters": True, "order": [self.provider],
                                "allow_fallbacks": False}
        body.update(self.extra_body)
        msgs = self._with_reasoning(messages)
        for _attempt in range(2):  # an empty reply: ask once more
            start = time.monotonic()
            data = self._post({**body, "messages": msgs})
            self.usage["seconds"] += time.monotonic() - start
            self.usage["calls"] += 1
            u = data.get("usage") or {}
            self.usage["prompt_tokens"] += u.get("prompt_tokens") or 0
            self.usage["completion_tokens"] += u.get("completion_tokens") or 0
            self.usage["billed_cost"] += u.get("cost") or 0.0
            prov = data.get("provider") or "?"
            self.usage["providers"][prov] = self.usage["providers"].get(prov, 0) + 1
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or "").replace("<think>", "").replace("</think>", "").strip()
            if msg.get("tool_calls") or text:
                break
            self.usage["empty"] = self.usage.get("empty", 0) + 1
            msgs = [*msgs, {"role": "user", "content":
                            "(Write your next turn now: a tool call, or a message to the customer.)"}]
        calls = [{"id": tc["id"], "type": "function",
                  "function": {"name": tc["function"]["name"],
                               "arguments": tc["function"].get("arguments") or "{}"}}
                 for tc in (msg.get("tool_calls") or [])]
        if msg.get("reasoning_details"):
            self.reasoning[calls[0]["id"] if calls else text] = msg["reasoning_details"]
        if calls:  # τ²-bench's rule: a tool call or a message, not both
            return {"role": "assistant", "content": None, "tool_calls": calls}
        if not text:
            raise EmptyTeacherReply(f"{self.slug} returned an empty reply")
        return {"role": "assistant", "content": text}

    def usage_dict(self) -> dict:
        d = super().usage_dict()
        if self.usage["billed_cost"]:  # OpenRouter's own bill beats our price × tokens
            d["est_cost"] = round(self.usage["billed_cost"], 4)
        return d


class AyaAgent:
    """A Cohere Aya model, prompted tool calls in Aya's own format (default: the 32B teacher)."""

    def __init__(self, model: str = "c4ai-aya-expanse-32b", prices: tuple[float, float] | None = None,
                 **server):  # api_base / api_key for a self-hosted model
        self.client = TeacherClient("prompted", style="cohere", model=model, prices=prices, **server)
        self.model = f"cohere/{self.client.model}"

    def turn(self, messages, tools):
        return self.client.turn(messages, tools)

    def usage_dict(self) -> dict:
        return self.client.usage.as_dict()


def make_agent(name: str) -> Callable[[], Any]:
    from reflexive.llm import default_model

    if name == "aya":
        return AyaAgent
    if name == "qwen":
        return lambda: NativeAgent("openrouter/qwen/qwen3-32b", PRICES["qwen"], no_think=True)
    if name == "gpt":
        return lambda: NativeAgent(default_model(), PRICES["gpt"])
    if name == "tiny":
        return lambda: AyaAgent("tiny-aya-global", prices=(0.0, 0.0))  # price not published
    if name == "aya-steps":
        return AyaAgent
    if name in VARIANTS:  # a slate model with a different request setting, e.g. thinking off
        slug, extra = VARIANTS[name]
        info = openrouter_info(slug)

        def variant():
            agent = OpenRouterAgent(slug, info["prices"], supported=info["supported"])
            agent.extra_body = dict(extra)
            return agent

        return variant
    if name in SLATE or name.startswith("or:"):
        slug = SLATE.get(name) or name[3:]
        info = openrouter_info(slug)
        return lambda: OpenRouterAgent(slug, info["prices"], supported=info["supported"])
    if name == "aya8":  # self-hosted: Cohere retired c4ai-aya-expanse-8b on 4 Apr 2026
        return _self_hosted_aya8
    raise ValueError(name)


class ModelUnavailable(RuntimeError):
    """The model can't be reached at all (removed, wrong ID, server not set): stop the run."""


def _self_hosted_aya8() -> AyaAgent:
    import os

    base, model = os.environ.get("AYA8_API_BASE"), os.environ.get("AYA8_MODEL")
    if not base or not model:
        raise ModelUnavailable(
            "Aya Expanse 8B is no longer on Cohere's API (removed 4 Apr 2026). Serve it yourself "
            "and set AYA8_API_BASE and AYA8_MODEL; see docs/TEACHER_SETUP.md.")
    return AyaAgent(model, prices=(0.0, 0.0),  # self-hosted: no API price
                    api_base=base, api_key=os.environ.get("AYA8_API_KEY", "none"))

# (max_context, reserve) per model: the student gets its real 8K window, the others don't need one
CONTEXT = {"tiny": (8192, 1024), "aya8": (8192, 1024)}
# characters ÷ 4 undercounted Tiny Aya's tokenizer (3 Nov: Cohere refused a prompt by 35 tokens)
COUNT = {"tiny": lambda msgs: int(estimate_tokens(msgs) * 1.15),
         "aya8": lambda msgs: int(estimate_tokens(msgs) * 1.15)}
# τ²-bench resolution-step hints (train tasks only; see Tau2Episode(hints=...))
HINTS = {"aya-steps": "steps"}
MODELS = ["aya", "qwen", "gpt", "tiny", "aya8", "aya-steps", *SLATE, *VARIANTS]


@functools.cache
def _openrouter_models() -> dict[str, dict]:
    """OpenRouter's public model list (no key needed), cached in results/ for a day."""
    cache = pathlib.Path("results/openrouter_models.json")
    if not cache.exists() or time.time() - cache.stat().st_mtime > 86_400:
        try:
            import httpx

            data = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data))
        except Exception as e:  # noqa: BLE001 - offline: fall back to the cache or fixed prices
            print(f"   (couldn't fetch OpenRouter's model list: {type(e).__name__})")
    return {m["id"]: m for m in json.loads(cache.read_text())} if cache.exists() else {}


def openrouter_info(slug: str) -> dict:
    """Listed price per token, open weights or not (has a Hugging Face link), context length."""
    m = _openrouter_models().get(slug, {})
    p = m.get("pricing") or {}
    fixed = next((PRICES[k] for k, v in SLATE.items() if v == slug and k in PRICES), (0.0, 0.0))
    prices = (float(p["prompt"]), float(p["completion"])) if "prompt" in p else fixed
    return {"prices": prices, "open_weights": bool(m.get("hugging_face_id")) if m else None,
            "hf": m.get("hugging_face_id"), "context": m.get("context_length"), "listed": bool(m),
            "supported": set(m.get("supported_parameters") or []) or None}


def resolve_models(names: list[str]) -> list[str]:
    out = []
    for n in names:
        out += list(SLATE) if n == "bakeoff" else [n]
    return list(dict.fromkeys(out))


def _file_name(model: str) -> str:
    return model.replace("or:", "or-").replace("/", "__")


# --- One conversation ---------------------------------------------------------------------------


def run_one(model: str, domain: str, task_id: str, *, temperature: float = 0.0,
            judges: bool = False, provider: str | None = None, user_llm: str | None = None,
            user_llm_args: dict | None = None) -> dict:
    agent = make_agent(model)()
    if hasattr(agent, "temperature"):
        agent.temperature = temperature
    if provider:
        agent.provider = provider
    max_context, reserve = CONTEXT.get(model, (128_000, 4_096))
    ep = Tau2Episode(domain, task_id, user_llm=user_llm, user_llm_args=user_llm_args,
                     max_context=max_context, reserve=reserve,
                     count_tokens=COUNT.get(model, estimate_tokens), hints=HINTS.get(model, "none"))
    start = time.monotonic()
    error = None
    try:
        ep.reset()
        while not ep.done:
            messages = ep.prompt()
            if ep.done:
                break
            try:
                ep.step(agent.turn(messages, ep.tool_schemas))
            except ContextTooLong:  # the API's own token count was over the window: an overflow
                ep._end_for_context()
    except Exception as e:  # noqa: BLE001 - one bad conversation shouldn't stop the sample
        error = f"{type(e).__name__}: {str(e)[:300]}"
        if ep._env is not None and not ep.done:
            ep.finish()
    if not ep.done:  # failed before the conversation started: record it as a failed attempt
        rec = {"domain": domain, "task_id": task_id, "termination": "unexpected_error",
               "outcome": {"passed": False, "reward": 0.0, "checks": {}, "judged": False,
                           "skipped": [], "note": error}, "messages": [], "user_cost": None,
               "sample_model": model, "agent_llm": agent.model,
               "agent_usage": agent.usage_dict(), "error": error}
    else:
        meta = {"sample_model": model, "agent_llm": agent.model, "agent_usage": agent.usage_dict(),
                "error": error, "temperature": temperature, "pinned_provider": provider}
        try:
            rec = ep.record(judges=judges, **meta)
        except Exception as e:  # noqa: BLE001 - a judge call failed: keep the run, re-judge later
            if not judges:
                raise
            rec = ep.record(judges=False, **meta, judge_error=f"{type(e).__name__}: {str(e)[:300]}")
    msgs = rec["messages"]
    rec["diagnostics"] = {
        "tools_ran": sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant"),
        "unseen_arguments": unseen_arguments(msgs),
        **tool_errors(msgs),
    }
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec


# --- Summary ------------------------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence range for a pass rate of k out of n (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


BROKEN_ENDINGS = {"unexpected_error", "infrastructure_error", "agent_error", "user_error"}


def _broken(rec: dict) -> bool:
    """The run broke (an API failure, a crash), so it says nothing about the model's ability."""
    return bool(rec.get("error")) or rec.get("termination") in BROKEN_ENDINGS


def summarise(model: str, recs: list[dict]) -> dict:
    n = len(recs)
    k = sum(r["outcome"]["passed"] for r in recs)
    lo, hi = wilson(k, n)
    diag = [r["diagnostics"] for r in recs]
    by_domain = {d: (sum(r["outcome"]["passed"] for r in recs if r["domain"] == d),
                     sum(1 for r in recs if r["domain"] == d)) for d in SAMPLE}
    return {
        "model": model, "n": n, "passed": k, "rate": k / n if n else 0.0, "ci": (lo, hi),
        "by_domain": by_domain,
        "invented_ids": sum(bool(x["unseen_arguments"]) for x in diag),
        "unknown_tools": sum(bool(x["unknown_tools"]) for x in diag),
        "repeated_failures": sum(bool(x["repeated_failures"]) for x in diag),
        "no_tools": sum(x["tools_ran"] == 0 for x in diag),
        "overflow": sum(bool((r.get("flags") or {}).get("context_overflow")) for r in recs),
        "errors": sum(_broken(r) for r in recs),
        "cost": sum(r["agent_usage"]["est_cost"] + (r["user_cost"] or 0) for r in recs),
        "agent_cost": sum(r["agent_usage"]["est_cost"] for r in recs),
        "open_weights": _open_weights(model),
        "mean_s": sum(r["wall_s"] for r in recs) / n if n else 0.0,
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== Summary: 30 train tasks that need a state change (DB checks only: a best case) ===")
    print(f"{'model':<9} {'open':>4} {'passed':>9} {'95% range':>12} {'retail':>7} {'airline':>8} "
          f"{'invented IDs':>13} {'unknown tool':>13} {'retried fail':>13} {'no tools':>9} "
          f"{'over 8K':>8} {'errors':>7} {'cost':>7} {'$/pass':>7}")
    for s in rows:
        if not s["n"]:
            print(f"{s['model'][:9]:<9}  (no finished runs)")
            continue
        lo, hi = s["ci"]
        dom = s["by_domain"]
        ow = {True: "yes", False: "no", None: "?"}[s["open_weights"]]
        per = f"${s['agent_cost'] / s['passed']:.3f}" if s["passed"] else "-"
        print(f"{s['model'][:9]:<9} {ow:>4} {s['passed']:>3}/{s['n']:<3} ({s['rate']:>4.0%}) "
              f"{lo:>4.0%}–{hi:<4.0%} {dom['retail'][0]:>3}/{dom['retail'][1]:<3} "
              f"{dom['airline'][0]:>3}/{dom['airline'][1]:<4} "
              f"{s['invented_ids']:>9}/{s['n']:<3} {s['unknown_tools']:>9}/{s['n']:<3} "
              f"{s['repeated_failures']:>9}/{s['n']:<3} {s['no_tools']:>5}/{s['n']:<3} "
              f"{s['overflow']:>4}/{s['n']:<3} {s['errors']:>7} ${s['cost']:>6.2f} {per:>7}")
    print("Counts in the hallucination columns are conversations with at least one such event.")
    print("open = weights published (OpenRouter links them on Hugging Face); $/pass = the agent's "
          "own cost per passed task (customer excluded).")
    if any(s["model"] == "tiny" for s in rows):
        print("tiny: cost is the simulated customer only (Tiny Aya's API price isn't published); "
              "'over 8K' = ended by the context guard (estimated tokens).")
    if any(s["errors"] for s in rows):
        print("⚠ 'errors' are broken runs (API failures, crashes): they count as fails above but say "
              "nothing about the model. Fix the cause and re-run with --retry-broken.")
    aya = next((s for s in rows if s["model"] == "aya"), None)
    refs = [s for s in rows if s["model"] != "aya"]
    if aya and refs:
        print("\nDecision rule (agreed before running):")
        print("  Aya ≤ 3/30 while a reference ≥ ~15/30  → unaided Aya is a poor teacher; test hints next.")
        print("  Aya ≥ ~9/30 with few invented IDs      → usable teacher; best-of-N will give enough fixes.")
        print("  In between                             → add ~20 tasks, or go straight to the hinted test.")


STUDENTS = ("aya", "aya8", "tiny")  # candidate students; every other model is a candidate teacher


def _open_weights(model: str) -> bool | None:
    if model in VARIANTS:
        return openrouter_info(VARIANTS[model][0])["open_weights"]
    if model in SLATE or model.startswith("or:"):
        return openrouter_info(SLATE.get(model) or model[3:])["open_weights"]
    return {"aya": True, "aya-steps": True, "tiny": True, "aya8": True, "qwen": True,
            "gpt": False}.get(model)


def _no_credits(rec: dict) -> bool:
    """Truly out of credits. OpenRouter's in-flight budget (too much reserved by requests
    already running) is temporary, so it doesn't count: those conversations are retried."""
    err = rec.get("error") or ""
    if "in_flight" in err or "in-flight" in err:
        return False
    return "OutOfCredits" in err or "402" in err or "available credits" in err


def _gone(rec: dict) -> bool:
    err = rec.get("error") or ""
    return "NotFoundError" in err or "was removed" in err or "ModelUnavailable" in err


def print_overlap(passed_by: dict[str, set]) -> None:
    """Teacher vs student: the tasks only the teacher solves are the possible fixes."""
    teachers = sorted((m for m in passed_by if m not in STUDENTS or m == "aya"),
                      key=lambda m: -len(passed_by[m]))
    pairs = [(t, s) for t in teachers for s in STUDENTS if t != s and t in passed_by and s in passed_by]
    if not pairs:
        return
    print("\nPossible teacher fixes (tasks the teacher solves and the student doesn't):")
    for t, s in pairs:
        teacher, student = passed_by[t], passed_by[s]
        fix = sorted(teacher - student)
        print(f"  {t[:9]:>9} → {s:<5} {len(fix):>2}   ({len(student - teacher)} the other way, "
              f"{len(teacher & student)} both)  {', '.join(f'{d}:{i}' for d, i in fix) or '-'}")


# --- Main ---------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["aya", "qwen"],
                    help=f"any of {', '.join(MODELS)}; 'bakeoff' for the whole slate; or:<openrouter slug>")
    ap.add_argument("--json", type=pathlib.Path, help="also write the summary rows to this file")
    ap.add_argument("--workers", type=int, default=4, help="conversations run at once per model")
    ap.add_argument("--fresh", action="store_true", help="ignore earlier results and start again")
    ap.add_argument("--list", action="store_true", help="only print the task sample")
    ap.add_argument("--limit", type=int, help="run only the first N tasks (a quick smoke test)")
    ap.add_argument("--retry-broken", action="store_true",
                    help="re-run conversations that broke (API failures, crashes)")
    a = ap.parse_args()
    a.models = resolve_models(a.models)
    bad = [m for m in a.models if m not in MODELS and not m.startswith("or:")]
    if bad:
        ap.error(f"unknown model(s) {bad}; choose from {MODELS}, 'bakeoff' or or:<slug>")

    tasks = sample_tasks()  # also loads each domain's tasks once, before the worker threads
    tasks = tasks[:a.limit] if a.limit else tasks
    from tau2.gym import register_gym_agent

    import reflexive.episode as episode

    register_gym_agent()  # once, here, rather than raced by the workers
    episode._REGISTERED = True
    if a.list:
        for d, t in tasks:
            print(d, t)
        return 0
    rows = []
    passed_by: dict[str, set] = {}
    stop_all = False
    for model in a.models:
        if stop_all:
            break
        out = pathlib.Path(f"results/episodes/sample-{_file_name(model)}.jsonl")
        if model in SLATE or model.startswith("or:"):
            info = openrouter_info(SLATE.get(model) or model[3:])
            if info["listed"] is False and _openrouter_models():
                print(f"\n== {model}: ⛔ not on OpenRouter's model list, skipped")
                continue
        out.parent.mkdir(parents=True, exist_ok=True)
        done: dict[tuple[str, str], dict] = {}
        if out.exists() and not a.fresh:
            for line in out.read_text().splitlines():
                r = json.loads(line)
                if a.retry_broken and _broken(r):
                    continue
                done[(r["domain"], r["task_id"])] = r
        todo = [t for t in tasks if t not in done]
        print(f"\n== {model}: {len(tasks)} tasks ({len(done)} already done) → {out}")
        if a.retry_broken and not a.fresh:
            out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))
        uses_openrouter = model in SLATE or model.startswith("or:") or model in ("qwen", "gpt")
        left = openrouter_credits() if todo and uses_openrouter else None
        if left is not None:
            print(f"   OpenRouter credits left: ${left:.2f}")
            if left < MIN_CREDITS:
                print(f"   ⛔ under ${MIN_CREDITS:.0f}: add credits at "
                      "https://openrouter.ai/settings/credits, then run the same command again")
                break
        try:
            if todo:
                make_agent(model)()  # fail now, not 30 times, if the model can't be set up
        except ModelUnavailable as e:
            print(f"   ⛔ {e}")
            continue
        with out.open("w" if a.fresh else "a") as f, ThreadPoolExecutor(a.workers) as pool:
            futures = {pool.submit(run_one, model, d, t): (d, t) for d, t in todo}
            for fut in as_completed(futures):
                d, t = futures[fut]
                rec = fut.result()
                if _no_credits(rec):  # every other call would fail too, for every model
                    for other in futures:
                        other.cancel()
                    print("   ⛔ OpenRouter is out of credits: stopping. Add credits, then run the same "
                          "command with --retry-broken (finished tasks are kept)")
                    stop_all = True
                    break
                if _gone(rec):  # removed model / wrong ID: every other task would fail the same way
                    for other in futures:
                        other.cancel()
                    print(f"   ⛔ {model}: model not available, stopping ({rec['error'][:160]})")
                    break
                f.write(json.dumps(rec) + "\n")
                f.flush()
                done[(d, t)] = rec
                x = rec["diagnostics"]
                flags = [s for s, on in (("invented IDs", x["unseen_arguments"]),
                                         ("unknown tool", x["unknown_tools"]),
                                         ("retried a failed call", x["repeated_failures"]),
                                         ("no tools ran", not x["tools_ran"]),
                                         ("over 8K", (rec.get("flags") or {}).get("context_overflow")),
                                         (f"BROKEN RUN ({rec['termination']}): {rec.get('error') or ''}",
                                          _broken(rec))) if on]
                print(f"   {'✅' if rec['outcome']['passed'] else '❌'} {d}:{t:<6} "
                      f"{x['tools_ran']:>2} tools  {rec['wall_s']:>5.0f}s  {'; '.join(flags)}")
        rows.append(summarise(model, [done[t] for t in tasks if t in done]))
        passed_by[model] = {t for t in tasks if t in done and done[t]["outcome"]["passed"]}
    print_summary(rows)
    print_overlap(passed_by)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
