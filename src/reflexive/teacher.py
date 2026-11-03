"""The teacher model, Aya Expanse 32B via Cohere's API, as an agent for `Tau2Episode` (SEL-37).

    teacher = TeacherClient(mode="prompted")        # or "native"; reads COHERE_API_KEY
    ep = Tau2Episode("retail", task_id, max_context=128_000)
    ep.reset()
    while not ep.done:
        ep.step(teacher.turn(ep.prompt(), ep.tool_schemas))
    teacher.usage                                   # calls, tokens, seconds, est. cost

Calls go through Cohere's OpenAI-compatible endpoint (https://api.cohere.ai/compatibility/v1)
with LiteLLM. See docs/TEACHER_SETUP.md for limits and prices.

**Two ways to make tool calls** (`scripts/teacher_probe.py` shows which one works):

- ``native``: pass the tool schemas as ``tools=`` and read structured ``tool_calls`` back.
  Cohere doesn't document whether Aya Expanse supports this.
- ``prompted``: list the tools in the system prompt and ask for a JSON tool call in the reply.
  This uses the **same format as AgentTune's fallback for models without a tool template**
  (which is what Tiny Aya gets): a JSON list ``[{"tool_name": ..., "parameters": {...}}]``, with
  results returned as a user turn ``Tool result for <name>: ...``. That keeps the teacher's
  fixes in the shape the student is trained on. The exact shared format is for the data
  contract (SEL-35) to confirm. This module's copy is ours, kept deliberately small.

Either way, `turn()` returns an OpenAI-style assistant dict that `Tau2Episode.step` accepts,
and the saved trajectory is always OpenAI-style, whichever mode produced it.

**Rate limiting.** Off by default: we use a production key. If a key does have a per-minute
cap (trial keys: 20/min and 1,000 calls a month), set ``rpm`` or the TEACHER_RPM environment
variable and calls are spaced to stay under it. Rate-limit errors are retried either way.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

COHERE_COMPAT_BASE = "https://api.cohere.ai/compatibility/v1"
TEACHER_MODEL = "c4ai-aya-expanse-32b"
# Listed prices (third-party listing, 3 Nov 2026; confirm on cohere.com/pricing): USD per token.
PRICE_IN, PRICE_OUT = 0.50 / 1e6, 1.50 / 1e6

# Same wording as AgentTune's DEFAULT_TOOLS_FALLBACK_PROMPT / DEFAULT_TOOL_RESULT_FORMAT, so the
# teacher's prompted transcripts match what the student sees in training.
TOOLS_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question as plain text."
)
TOOL_RESULT = "Tool result for {name}: {content}\nUse this result to answer the original question."


# --- Prompted tool-call format -----------------------------------------------------------------


def to_prompted(messages: list[dict], tool_schemas: list[dict]) -> list[dict]:
    """Rewrite an OpenAI-style conversation for a model without native tool calling.

    The tools go into the system prompt; earlier agent tool calls become their JSON-list text;
    tool results become ``Tool result for <name>: ...`` user turns. Adjacent turns from the same
    role are merged, so roles alternate.
    """
    names: dict[str, str] = {}
    out: list[dict] = []
    tools_text = json.dumps([s.get("function", s) for s in tool_schemas], separators=(",", ":"))
    for m in messages:
        role, content = m["role"], m.get("content")
        if role == "system":
            out.append({"role": "system",
                        "content": f"{content}\n\n{TOOLS_PROMPT.format(tools=tools_text)}"})
            continue
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
                names[tc.get("id", "")] = fn["name"]
                calls.append({"tool_name": fn["name"], "parameters": args})
            out.append({"role": "assistant", "content": json.dumps(calls)})
            continue
        if role == "tool":
            name = names.get(m.get("tool_call_id", ""), "tool")
            out.append({"role": "user", "content": TOOL_RESULT.format(name=name, content=content)})
            continue
        out.append({"role": role, "content": content or ""})
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"] and m["role"] != "system":
            merged[-1] = {**merged[-1], "content": f"{merged[-1]['content']}\n\n{m['content']}"}
        else:
            merged.append(m)
    return merged


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _first_json(text: str) -> Any:
    """The first JSON list or object in ``text`` (fences allowed), or None."""
    text = _FENCE.sub("", text.strip())
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "[{":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except ValueError:
                continue
    return None


def parse_prompted(text: str, id_prefix: str = "call") -> list[dict] | None:
    """Tool calls from a prompted reply, as OpenAI-style ``tool_calls``; None if it isn't one.

    Accepts the requested ``[{"tool_name", "parameters"}]`` list, a single object, and the
    common near-misses ``name`` / ``arguments``. A reply is only treated as a tool call if it
    *starts* with the JSON (after optional fences), so a message to the customer that merely
    mentions JSON stays a message. A well-formed call to a tool that doesn't exist is kept as a
    call: τ²-bench then records it, and it is exactly the Tool hallucination we want to detect.
    """
    stripped = _FENCE.sub("", text.strip())
    if not stripped.startswith(("[", "{")):
        return None
    value = _first_json(stripped)
    items = value if isinstance(value, list) else [value] if isinstance(value, dict) else None
    if not items:
        return None
    calls = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        name = item.get("tool_name") or item.get("name")
        args = item.get("parameters", item.get("arguments", {}))
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append({"id": f"{id_prefix}_{i}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}})
    return calls


# --- Client ----------------------------------------------------------------------------------


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (thread-safe)."""

    def __init__(self, rpm: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next:
                self._sleep(self._next - now)
                now = self._next
            self._next = now + self.interval


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    errors: int = 0
    unparsed: int = 0  # prompted replies that looked like a tool call but didn't parse
    per_call: list[dict] = field(default_factory=list)

    @property
    def cost(self) -> float:
        """Estimated USD at the listed prices (PRICE_IN / PRICE_OUT)."""
        return self.prompt_tokens * PRICE_IN + self.completion_tokens * PRICE_OUT

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "seconds": round(self.seconds, 2),
                "errors": self.errors, "unparsed": self.unparsed, "est_cost": round(self.cost, 4)}


class TeacherClient:
    """Aya Expanse 32B (by default) as the agent: one call per agent turn."""

    def __init__(self, mode: str = "prompted", *, model: str = TEACHER_MODEL,
                 api_base: str = COHERE_COMPAT_BASE, api_key: str | None = None,
                 rpm: float | None = None, temperature: float = 0.0, max_tokens: int = 1024,
                 seed: int | None = None, num_retries: int = 3,
                 completion: Callable[..., Any] | None = None):
        if mode not in ("native", "prompted"):
            raise ValueError("mode must be 'native' or 'prompted'")
        self.mode, self.model, self.api_base = mode, model, api_base
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key and completion is None:
            raise RuntimeError("Set COHERE_API_KEY in .env (see docs/TEACHER_SETUP.md)")
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.num_retries = num_retries
        rpm = rpm if rpm is not None else float(os.environ.get("TEACHER_RPM") or 0)  # 0 = no limit
        self.limiter = RateLimiter(rpm)
        self._completion = completion
        self.usage = Usage()

    def _call(self, messages: list[dict], tools: list[dict] | None) -> Any:
        if self._completion is None:
            import litellm

            self._completion = litellm.completion
        kwargs: dict[str, Any] = dict(
            model=f"openai/{self.model}", api_base=self.api_base, api_key=self.api_key,
            messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
            num_retries=self.num_retries,
        )
        if tools:
            kwargs["tools"] = tools
        if self.seed is not None:
            kwargs["seed"] = self.seed
        self.limiter.wait()
        start = time.monotonic()
        try:
            resp = self._completion(**kwargs)
        except Exception:
            self.usage.errors += 1
            raise
        finally:
            self.usage.seconds += time.monotonic() - start
        self.usage.calls += 1
        u = getattr(resp, "usage", None)
        pt = getattr(u, "prompt_tokens", 0) or 0
        ct = getattr(u, "completion_tokens", 0) or 0
        self.usage.prompt_tokens += pt
        self.usage.completion_tokens += ct
        self.usage.per_call.append({"prompt_tokens": pt, "completion_tokens": ct})
        return resp

    def turn(self, messages: list[dict], tool_schemas: list[dict]) -> dict:
        """One agent turn for ``messages`` (from `Tau2Episode.prompt()`), as an assistant dict."""
        if self.mode == "native":
            msg = self._call(messages, tool_schemas).choices[0].message
            calls = [{"id": tc.id, "type": "function",
                      "function": {"name": tc.function.name,
                                   "arguments": tc.function.arguments or "{}"}}
                     for tc in (getattr(msg, "tool_calls", None) or [])]
            if calls:  # τ²-bench's rule: a tool call or a message, not both
                return {"role": "assistant", "content": None, "tool_calls": calls}
            return {"role": "assistant", "content": msg.content or ""}

        msg = self._call(to_prompted(messages, tool_schemas), None).choices[0].message
        text = msg.content or ""
        calls = parse_prompted(text, id_prefix=f"t{self.usage.calls}")
        if calls:
            return {"role": "assistant", "content": None, "tool_calls": calls}
        if text.strip().startswith(("[", "{")):
            self.usage.unparsed += 1  # looked like a call but wasn't valid: sent as a message
        return {"role": "assistant", "content": text}
