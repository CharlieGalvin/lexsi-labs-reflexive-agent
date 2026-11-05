"""Run one τ²-bench conversation with an API model as the agent, and record it (SEL-58, SEL-45).

The teacher bake-off (``scripts/teacher/bakeoff.py``) and the teacher fix run
(``scripts/teacher/fixes.py``) both call ``run_one``. Model names are short keys:

- ``aya``: Aya Expanse 32B via Cohere, prompted tool calls in Aya's own format (our student).
- ``tiny``: Tiny Aya (``tiny-aya-global``) via Cohere, prompted the same way, inside its real
  8,192-token window. Its API price isn't published, so its cost is the simulated customer only.
- ``qwen``: Qwen3 32B via OpenRouter, native tools, thinking off: phase 1's same-size reference.
- The **bake-off slate** (``SLATE``) and its request-setting ``VARIANTS``: OpenRouter models
  with native tool calling. Any other OpenRouter model runs as ``or:<slug>``. Prices and
  whether the weights are open come live from OpenRouter's public model list.

API clients (litellm, httpx) are imported only when a model is called, so the offline tests
and CI need neither.
"""

from __future__ import annotations

import functools
import json
import os
import pathlib
import random
import time
from collections.abc import Callable
from typing import Any

from reflexive.episode import Tau2Episode, estimate_tokens, official_split
from reflexive.teacher import (
    CohereClient,
    ContextTooLong,
    EmptyTeacherReply,
    tool_errors,
    unseen_arguments,
)

Record = dict[str, Any]
Message = dict[str, Any]
Task = tuple[str, str]  # (domain, task_id)

# --- Models -------------------------------------------------------------------------------------

# USD per token, for models whose cost LiteLLM doesn't report (listed prices, 3 Nov 2026)
PRICES = {"qwen": (0.08e-6, 0.28e-6), "glm": (0.07e-6, 0.40e-6)}

# Teacher bake-off (SEL-58): OpenRouter slugs. Picked from OpenRouter's τ²-bench airline board
# (3 Nov 2026) to span open vs closed weights and cheap vs strong; scores are that board's.
SLATE = {
    # open weights (expected; the run checks OpenRouter's Hugging Face link)
    "glm": "z-ai/glm-4.7-flash",  # 66.7%, our first pick
    "glm53f": "z-ai/glm-5.3-flash",  # 75.6%, the chosen teacher
    "dsv4f": "deepseek/deepseek-v4-flash",  # 73.9%
    "mimo": "xiaomi/mimo-v2.6-flash",  # 77.9%
    "qwen38": "qwen/qwen3.8-27b",  # 77.5%, dense 27B
    "gemma4": "google/gemma-4-31b-it",  # 75.3%, dense 31B
    "gptoss": "openai/gpt-oss-120b",  # 65.2%
    # closed weights
    "haiku": "anthropic/claude-haiku-5.5",  # 72.8%
    # gflite (google/gemini-3.5-flash-lite) dropped 3 Nov at Charlie's request; or:<slug> still runs it
    "sonnet": "anthropic/claude-sonnet-5.5",  # 78.8%
    "gpt51": "openai/gpt-5.1",  # 80.0%, the ceiling
    "gem38f": "google/gemini-3.8-flash",  # 71.3%, added 3 Nov at Charlie's request
    "opus": "anthropic/claude-opus-5.5",  # 75.3%, added 3 Nov; the priciest (~$0.34/task)
}

# Request-setting variants of slate models (not part of --models bakeoff)
VARIANTS = {
    "qwen38nt": ("qwen/qwen3.8-27b", {"reasoning": {"enabled": False}}),  # Qwen3.8 27B, thinking off
}

STUDENTS = ("aya", "tiny")  # candidate students; every other model is a candidate teacher
MODELS = ["aya", "tiny", "qwen", *SLATE, *VARIANTS]

# (max_context, reserve) per model: Tiny Aya gets its real 8K window, the others don't need one
CONTEXT = {"tiny": (8192, 1024)}
# characters ÷ 4 undercounted Tiny Aya's tokenizer (3 Nov: Cohere refused a prompt by 35 tokens)
COUNT: dict[str, Callable[[list[Message]], int]] = {"tiny": lambda msgs: int(estimate_tokens(msgs) * 1.15)}

MIN_CREDITS = 2.0  # USD; below this a model's 30 conversations can run dry halfway
EMPTY_NUDGE = "(Write your next turn now: a tool call, or a message to the customer.)"


class ModelError(RuntimeError):
    """The provider failed mid-reply (finish_reason 'error'), even after retries."""


class OutOfCredits(RuntimeError):
    """OpenRouter refused for lack of credits (HTTP 402): stop the whole run."""


class ModelUnavailable(RuntimeError):
    """The model can't be reached at all (removed, wrong ID): stop the run."""


def _strip_think(text: str | None) -> str:
    return (text or "").replace("<think>", "").replace("</think>", "").strip()


# --- Agents -------------------------------------------------------------------------------------


class NativeAgent:
    """An API model with native tool calling, as the agent (one call per turn), via LiteLLM."""

    def __init__(
        self,
        model: str,
        prices: tuple[float, float],
        no_think: bool = False,
        supported: set[str] | None = None,
    ) -> None:
        self.model, self.prices, self.no_think = model, prices, no_think
        # OpenRouter's list of parameters the model accepts. Some newer models (reasoning ones)
        # refuse `temperature`; with require_parameters on, sending it finds no endpoint at all.
        self.supported = supported
        self.temperature = 0.0  # fix runs raise it for retries (scripts/teacher/fixes.py)
        self.usage: dict[str, Any] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def turn(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        import litellm

        litellm.suppress_debug_info = True  # ty: ignore[invalid-assignment]  # litellm types it Literal[False]
        if self.no_think and messages and messages[0]["role"] == "system":
            messages = [{**messages[0], "content": messages[0]["content"] + "\n\n/no_think"}, *messages[1:]]
        for _attempt in range(2):  # Qwen3 sometimes returns an empty reply: ask once more
            start = time.monotonic()
            extra: dict[str, Any] = (
                {"extra_body": {"provider": {"require_parameters": True}}}
                if self.model.startswith("openrouter/")
                else {}
            )  # tools must work
            if not self.supported or "temperature" in self.supported:
                extra["temperature"] = self.temperature
            resp = litellm.completion(
                model=self.model, messages=messages, tools=tools, num_retries=3, **extra
            )
            self.usage["seconds"] += time.monotonic() - start
            self.usage["calls"] += 1
            u = getattr(resp, "usage", None)
            self.usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
            msg = resp.choices[0].message
            if msg.tool_calls or _strip_think(msg.content):
                break
            self.usage["empty"] = self.usage.get("empty", 0) + 1
            messages = [*messages, {"role": "user", "content": EMPTY_NUDGE}]
        calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
            }
            for tc in (msg.tool_calls or [])
        ]
        if calls:  # τ²-bench's rule: a tool call or a message, not both
            return {"role": "assistant", "content": None, "tool_calls": calls}
        text = _strip_think(msg.content)
        if not text:
            raise EmptyTeacherReply(f"{self.model} returned an empty reply")
        return {"role": "assistant", "content": text}

    def usage_dict(self) -> dict[str, Any]:
        pin, pout = self.prices
        return {
            **self.usage,
            "seconds": round(self.usage["seconds"], 2),
            "est_cost": round(self.usage["prompt_tokens"] * pin + self.usage["completion_tokens"] * pout, 4),
        }


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

    def __init__(self, slug: str, prices: tuple[float, float], supported: set[str] | None = None) -> None:
        super().__init__(f"openrouter/{slug}", prices, supported=supported)
        self.slug = slug
        self.reasoning: dict[str, Any] = {}  # first tool-call id or reply text -> reasoning_details
        self.provider: str | None = None
        self.extra_body: dict[str, Any] = {}  # added to every request (VARIANTS)
        self.usage.update(billed_cost=0.0, providers={})

    def _with_reasoning(self, messages: list[Message]) -> list[Message]:
        out = []
        for m in messages:
            if m.get("role") == "assistant":
                key = (m.get("tool_calls") or [{}])[0].get("id") or m.get("content")
                if key in self.reasoning:
                    m = {**m, "reasoning_details": self.reasoning[key]}
            out.append(m)
        return out

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
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

    def turn(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        body: dict[str, Any] = {
            "model": self.slug,
            "tools": tools,
            "provider": {"require_parameters": True},
            "usage": {"include": True},
            # a cap keeps OpenRouter's per-request reservation small (it holds
            # max_tokens × price while a request runs); turns are far shorter
            "max_tokens": 8192,
        }
        if not self.supported or "temperature" in self.supported:
            body["temperature"] = self.temperature
        if self.provider:  # pin one provider (and its quantisation) so the run is reproducible
            body["provider"] = {
                "require_parameters": True,
                "order": [self.provider],
                "allow_fallbacks": False,
            }
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
            text = _strip_think(msg.get("content"))
            if msg.get("tool_calls") or text:
                break
            self.usage["empty"] = self.usage.get("empty", 0) + 1
            msgs = [*msgs, {"role": "user", "content": EMPTY_NUDGE}]
        calls: list[dict[str, Any]] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"].get("arguments") or "{}",
                },
            }
            for tc in (msg.get("tool_calls") or [])
        ]
        if msg.get("reasoning_details"):
            self.reasoning[calls[0]["id"] if calls else text] = msg["reasoning_details"]
        if calls:  # τ²-bench's rule: a tool call or a message, not both
            return {"role": "assistant", "content": None, "tool_calls": calls}
        if not text:
            raise EmptyTeacherReply(f"{self.slug} returned an empty reply")
        return {"role": "assistant", "content": text}

    def usage_dict(self) -> dict[str, Any]:
        d = super().usage_dict()
        if self.usage["billed_cost"]:  # OpenRouter's own bill beats our price × tokens
            d["est_cost"] = round(self.usage["billed_cost"], 4)
        return d


class AyaAgent:
    """A Cohere Aya model with prompted tool calls in Aya's own format (default: Aya Expanse 32B)."""

    def __init__(
        self, model: str = "c4ai-aya-expanse-32b", prices: tuple[float, float] | None = None
    ) -> None:
        self.client = CohereClient("prompted", style="cohere", model=model, prices=prices)
        self.model = f"cohere/{self.client.model}"

    def turn(self, messages: list[Message], tools: list[dict[str, Any]]) -> Message:
        return self.client.turn(messages, tools)

    def usage_dict(self) -> dict[str, Any]:
        return self.client.usage.as_dict()


Agent = NativeAgent | AyaAgent


def make_agent(name: str) -> Callable[[], Agent]:
    """A factory for the agent called ``name`` (one agent per conversation: usage is per agent)."""
    if name == "aya":
        return AyaAgent
    if name == "tiny":
        return lambda: AyaAgent("tiny-aya-global", prices=(0.0, 0.0))  # price not published
    if name == "qwen":
        return lambda: NativeAgent("openrouter/qwen/qwen3-32b", PRICES["qwen"], no_think=True)
    if name in VARIANTS:  # a slate model with a different request setting, e.g. thinking off
        slug, extra = VARIANTS[name]
        info = openrouter_info(slug)

        def variant() -> OpenRouterAgent:
            agent = OpenRouterAgent(slug, info["prices"], supported=info["supported"])
            agent.extra_body = dict(extra)
            return agent

        return variant
    if name in SLATE or name.startswith("or:"):
        slug = SLATE.get(name) or name[3:]
        info = openrouter_info(slug)
        return lambda: OpenRouterAgent(slug, info["prices"], supported=info["supported"])
    raise ValueError(f"unknown model {name!r}; choose from {MODELS} or or:<openrouter slug>")


def resolve_models(names: list[str]) -> list[str]:
    """Expand ``bakeoff`` into the slate, keeping order and dropping repeats."""
    out = []
    for n in names:
        out += list(SLATE) if n == "bakeoff" else [n]
    return list(dict.fromkeys(out))


def file_name(model: str) -> str:
    """A model key as a file name: ``or:a/b`` → ``or-a__b``."""
    return model.replace("or:", "or-").replace("/", "__")


def uses_openrouter(model: str) -> bool:
    return model in SLATE or model in VARIANTS or model.startswith("or:") or model == "qwen"


# --- OpenRouter: model list, prices, credits ----------------------------------------------------


@functools.cache
def _openrouter_models() -> dict[str, dict[str, Any]]:
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


def openrouter_info(slug: str) -> dict[str, Any]:
    """Listed price per token, open weights or not (has a Hugging Face link), context length."""
    m = _openrouter_models().get(slug, {})
    p = m.get("pricing") or {}
    fixed = next((PRICES[k] for k, v in SLATE.items() if v == slug and k in PRICES), (0.0, 0.0))
    prices = (float(p["prompt"]), float(p["completion"])) if "prompt" in p else fixed
    return {
        "prices": prices,
        "open_weights": bool(m.get("hugging_face_id")) if m else None,
        "hf": m.get("hugging_face_id"),
        "context": m.get("context_length"),
        "listed": bool(m),
        "supported": set(m.get("supported_parameters") or []) or None,
    }


def open_weights(model: str) -> bool | None:
    """Whether the model's weights are published (None: unknown)."""
    if model in VARIANTS:
        return openrouter_info(VARIANTS[model][0])["open_weights"]
    if model in SLATE or model.startswith("or:"):
        return openrouter_info(SLATE.get(model) or model[3:])["open_weights"]
    return {"aya": True, "tiny": True, "qwen": True}.get(model)


def openrouter_credits() -> float | None:
    """Credits left on the OpenRouter account in USD, or None if it can't be checked."""
    import httpx

    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/credits",
            timeout=20,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
        )
        d = r.json()["data"]
        return float(d["total_credits"]) - float(d["total_usage"])
    except Exception:  # noqa: BLE001 - no key, offline, or the endpoint changed: don't block the run
        return None


# --- What a saved run says ----------------------------------------------------------------------

BROKEN_ENDINGS = {"unexpected_error", "infrastructure_error", "agent_error", "user_error"}


def is_broken(rec: Record) -> bool:
    """The run broke (an API failure, a crash), so it says nothing about the model's ability."""
    return bool(rec.get("error")) or rec.get("termination") in BROKEN_ENDINGS


def out_of_credits(rec: Record) -> bool:
    """Truly out of credits. OpenRouter's in-flight budget (too much reserved by requests
    already running) is temporary, so it doesn't count: those conversations are retried."""
    err = rec.get("error") or ""
    if "in_flight" in err or "in-flight" in err:
        return False
    return "OutOfCredits" in err or "402" in err or "available credits" in err


def model_gone(rec: Record) -> bool:
    """The model was removed or its ID is wrong: every other task would fail the same way."""
    err = rec.get("error") or ""
    return "NotFoundError" in err or "was removed" in err or "ModelUnavailable" in err


# --- Tasks and one conversation -----------------------------------------------------------------

SAMPLE = {"retail": 20, "airline": 10}  # phase 1's fixed sample
SEED = 0


def sample_tasks(sizes: dict[str, int] = SAMPLE, seed: int = SEED) -> list[Task]:
    """Phase 1's fixed sample: train tasks whose expected actions change state, per domain."""
    from reflexive.outcome import scorer

    out = []
    for domain, n in sizes.items():
        sc = scorer(domain)
        pool = sorted(t for t in official_split(domain, "train") if sc.write_action_indices(t))
        out += [(domain, t) for t in random.Random(seed).sample(pool, min(n, len(pool)))]
    return out


def _converse(agent: Agent, ep: Tau2Episode) -> str | None:
    """Drive the conversation to its end. Returns the error that stopped it, if any."""
    try:
        ep.reset()
        while not ep.done:
            messages = ep.prompt()
            if ep.done:
                break
            try:
                ep.step(agent.turn(messages, ep.tool_schemas))
            except ContextTooLong:  # the API's own token count was over the window: an overflow
                ep.end_for_context()
    except Exception as e:  # noqa: BLE001 - one bad conversation shouldn't stop the run
        if ep.has_env and not ep.done:
            ep.finish()
        return f"{type(e).__name__}: {str(e)[:300]}"
    return None


def _record(
    model: str,
    agent: Agent,
    ep: Tau2Episode,
    domain: str,
    task_id: str,
    error: str | None,
    *,
    judges: bool,
    temperature: float,
    provider: str | None,
) -> Record:
    if not ep.done:  # failed before the conversation started: record it as a failed attempt
        return {
            "domain": domain,
            "task_id": task_id,
            "termination": "unexpected_error",
            "outcome": {
                "passed": False,
                "reward": 0.0,
                "checks": {},
                "judged": False,
                "skipped": [],
                "note": error,
            },
            "messages": [],
            "user_cost": None,
            "sample_model": model,
            "agent_llm": agent.model,
            "agent_usage": agent.usage_dict(),
            "error": error,
        }
    meta = {
        "sample_model": model,
        "agent_llm": agent.model,
        "agent_usage": agent.usage_dict(),
        "error": error,
        "temperature": temperature,
        "pinned_provider": provider,
    }
    try:
        return ep.record(judges=judges, **meta)
    except Exception as e:  # noqa: BLE001 - a judge call failed: keep the run, re-judge later
        if not judges:
            raise
        return ep.record(judges=False, **meta, judge_error=f"{type(e).__name__}: {str(e)[:300]}")


def run_one(
    model: str,
    domain: str,
    task_id: str,
    *,
    temperature: float = 0.0,
    judges: bool = False,
    provider: str | None = None,
    user_llm: str | None = None,
    user_llm_args: dict[str, Any] | None = None,
) -> Record:
    """One conversation of ``model`` as the agent on a τ²-bench task, scored and recorded.

    Never raises for a model or API failure: the record says what went wrong (``error``,
    ``termination``), so a run of many conversations carries on.
    """
    agent = make_agent(model)()
    if isinstance(agent, NativeAgent):
        agent.temperature = temperature
    if provider and isinstance(agent, OpenRouterAgent):
        agent.provider = provider
    max_context, reserve = CONTEXT.get(model, (128_000, 4_096))
    ep = Tau2Episode(
        domain,
        task_id,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        max_context=max_context,
        reserve=reserve,
        count_tokens=COUNT.get(model, estimate_tokens),
    )
    start = time.monotonic()
    error = _converse(agent, ep)
    rec = _record(
        model, agent, ep, domain, task_id, error, judges=judges, temperature=temperature, provider=provider
    )
    msgs = rec["messages"]
    rec["diagnostics"] = {
        "tools_ran": sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant"),
        "unseen_arguments": unseen_arguments(msgs),
        **tool_errors(msgs),
    }
    if ep.has_env:  # what the agent was actually shown (system prompt included)
        try:
            rec["agent_view"] = ep.messages()
        except Exception:  # noqa: BLE001 - never lose a conversation over this extra
            pass
    raw = getattr(getattr(agent, "client", None), "raw_turns", None)
    if raw:  # prompted models (Aya): the raw replies, to review how calls were written and read
        rec["raw_turns"] = raw
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec
