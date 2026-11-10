"""Aya Expanse 32B via Cohere's API as an agent for `Tau2Episode`: our student (SEL-37).

    aya = CohereClient(mode="prompted")             # or "native"; reads COHERE_API_KEY
    ep = Tau2Episode("retail", task_id, max_context=128_000)
    ep.reset()
    while not ep.done:
        ep.step(aya.turn(ep.prompt(), ep.tool_schemas))
    aya.usage                                       # calls, tokens, seconds, est. cost

(The module is called ``teacher`` because Aya was first tried as the teacher; GLM-5.3-Flash
won the bake-off, and Aya is now the student. ``TeacherClient`` is kept as an alias.)

Calls go through Cohere's OpenAI-compatible endpoint (https://api.cohere.ai/compatibility/v1)
with LiteLLM. See docs/TEACHER_SETUP.md for limits and prices.

**Two ways to make tool calls** (`scripts/archive/teacher_probe.py` shows which one works):

- ``native``: pass the tool schemas as ``tools=`` and read structured ``tool_calls`` back.
  **Aya Expanse doesn't support this:** Cohere rejects it with "tool use is not supported by
  the provided model" (probe, 3 Nov). Kept for other teacher models.
- ``prompted`` (default, and what Aya Expanse needs): list the tools in the system prompt
  and ask for a JSON tool call in the reply. Two wordings (``style``):

  - ``cohere`` (default): **Aya Expanse's own trained format**, Cohere's Command R tool-use
    prompt (as in the model's tool_use chat template, which Ollama also uses). Tools as Python
    stubs; calls as ``Action: ```json [{"tool_name", "parameters"}]``` ``; results in
    ``<results>`` tags; a ``directly-answer`` pseudo-tool for "no tool needed".
  - ``agenttune``: AgentTune's fallback for models without a tool template (what Tiny Aya
    gets): a bare JSON list, results as ``Tool result for <name>: ...``.

  Both use the same ``{"tool_name", "parameters"}`` objects, and the saved trajectory is
  OpenAI-style either way, so fixes convert to the student's format. Generation stops where
  Aya would start writing tool results itself (``STOP_SEQUENCES``), and a call written inside
  prose is still found (``find_prompted_calls``).

Either way, `turn()` returns an OpenAI-style assistant dict that `Tau2Episode.step` accepts,
and the saved trajectory is always OpenAI-style, whichever mode produced it.

**Rate limiting.** Off by default: we use a production key. If a key does have a per-minute
cap (trial keys: 20/min and 1,000 calls a month), set ``rpm`` or the TEACHER_RPM environment
variable and calls are spaced to stay under it. Rate limits, server errors and timeouts
are retried (up to ``num_retries``, with back-off); a rejected request is not.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
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

# Aya Expanse's own tool format: Cohere's Command R tool-use prompt (as in the model's tool_use
# chat template, which Ollama also uses): tools as Python stubs under "## Available Tools", and
# "Action:" followed by a JSON list of {"tool_name", "parameters"}. The last paragraph is ours:
# it adapts the format to τ²-bench, where a turn is a tool call *or* a message to the customer.
COHERE_TOOLS_PROMPT = """## Available Tools
Here is a list of tools that you have available to you:

```python
{stubs}
```

## Tool calls
Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to \
produce a good response to the user's last input. You can use any of the supplied tools any number \
of times, but you should aim to execute the minimum number of necessary actions for the input. \
You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of \
actions you want to call should be formatted as a list of json objects, for example:
```json
[
    {{
        "tool_name": title of the tool in the specification,
        "parameters": a dict of parameters to input into the tool as they are defined in the specs, \
or {{}} if it takes no parameters
    }}
]
```

In this conversation each of your turns is EITHER an Action OR a message to the customer, never \
both. After an Action, stop: the tool results come back to you in the next turn inside <results> \
tags. Never write tool results yourself, and never tell the customer an action has happened \
until its result says so. To reply to the customer, write the reply as plain text with no Action."""
COHERE_RESULT = "<results>\nDocument: {i}\ntool_name: {name}\nresult: {content}\n</results>"
# Stop the reply where Aya would start writing tool results itself (seen 3 Nov).
STOP_SEQUENCES = ["<results>", "Tool result", "Document: 0"]
DIRECTLY_ANSWER = {"directly-answer", "directly_answer"}
_PY_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
    "null": "None",
}


def _py_type(schema: dict) -> str:
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(_PY_TYPES.get(x, "Any") for x in t)
    if t == "array" and isinstance(schema.get("items"), dict):
        return f"list[{_py_type(schema['items'])}]"
    if "anyOf" in schema:
        return " | ".join(_py_type(x) for x in schema["anyOf"])
    return _PY_TYPES.get(t, "Any")


def tool_stub(schema: dict) -> str:
    """One tool as a Python function stub with a docstring, as Cohere's tool template renders it."""
    fn = schema.get("function", schema)
    params = fn.get("parameters") or {}
    props, required = params.get("properties") or {}, set(params.get("required") or [])
    sig = ", ".join(f"{k}: {_py_type(v)}" + ("" if k in required else " = None") for k, v in props.items())
    desc = "\n    ".join(line.strip() for line in (fn.get("description") or "").strip().splitlines())
    lines = [f"def {fn['name']}({sig}) -> list[dict]:", '    """' + desc]
    if props:
        lines += ["", "    Args:"]
        for k, v in props.items():
            opt = "" if k in required else ", optional"
            lines.append(f"        {k} ({_py_type(v)}{opt}): {(v.get('description') or '').strip()}")
    lines += ['    """', "    pass"]
    return "\n".join(lines)


# --- Prompted tool-call format -----------------------------------------------------------------


def to_prompted(messages: list[dict], tool_schemas: list[dict], style: str = "agenttune") -> list[dict]:
    """Rewrite an OpenAI-style conversation for a model without native tool calling.

    The tools go into the system prompt (after τ²-bench's instructions and policy, unchanged);
    earlier agent tool calls become their JSON-list text; tool results become user turns.
    Adjacent turns from the same role are merged, so roles alternate.

    - ``style="agenttune"``: AgentTune's fallback wording (what Tiny Aya gets); tools as JSON
      schemas; results as ``Tool result for <name>: ...``.
    - ``style="cohere"``: Aya Expanse's own trained format; tools as Python stubs; calls as
      ``Action: ```json [...]```  ``; results in ``<results>`` tags.
    """
    if style not in ("agenttune", "cohere"):
        raise ValueError("style must be 'agenttune' or 'cohere'")
    names: dict[str, str] = {}
    out: list[dict] = []
    if style == "cohere":
        tools_block = COHERE_TOOLS_PROMPT.format(stubs="\n\n".join(tool_stub(t) for t in tool_schemas))
    else:
        tools_text = json.dumps([t.get("function", t) for t in tool_schemas], separators=(",", ":"))
        tools_block = TOOLS_PROMPT.format(tools=tools_text)
    n_results = 0
    for m in messages:
        role, content = m["role"], m.get("content")
        if role == "system":
            out.append({"role": "system", "content": f"{content}\n\n{tools_block}"})
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
            text = json.dumps(calls)
            if style == "cohere":
                text = f"Action: ```json\n{json.dumps(calls, indent=4)}\n```"
            out.append({"role": "assistant", "content": text})
            continue
        if role == "tool":
            name = names.get(m.get("tool_call_id", ""), "tool")
            if style == "cohere":
                out.append(
                    {"role": "user", "content": COHERE_RESULT.format(i=n_results, name=name, content=content)}
                )
                n_results += 1
            else:
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


def _calls_from(value: Any, id_prefix: str, strict: bool) -> list[dict] | None:
    """OpenAI-style tool_calls from a parsed JSON value, or None if it isn't a tool call.

    ``strict`` (used for JSON found *inside* prose) only accepts the requested ``tool_name`` key,
    so an unrelated JSON snippet in a message to the customer isn't mistaken for a call.
    """
    items = value if isinstance(value, list) else [value] if isinstance(value, dict) else None
    if not items:
        return None
    calls = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            return None
        name = item.get("tool_name") if strict else item.get("tool_name") or item.get("name")
        args = item.get("parameters", item.get("arguments", {}))
        if not isinstance(name, str) or not isinstance(args, dict):
            return None
        calls.append(
            {
                "id": f"{id_prefix}_{i}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        )
    return calls


def _json_values(text: str) -> Iterator[Any]:
    """Every JSON list or object that starts somewhere in ``text``, in order."""
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] in "[{":
            try:
                value, end = decoder.raw_decode(text[i:])
                yield value
                i += end
                continue
            except ValueError:
                pass
        i += 1


def find_prompted_calls(text: str, id_prefix: str = "call") -> tuple[list[dict] | None, bool]:
    """Tool calls in a prompted reply, and whether they were embedded in other text.

    Returns ``(calls, embedded)``; ``calls`` is None if the reply isn't a tool call.

    - **Reply is the JSON** (as asked, optionally fenced): the requested
      ``[{"tool_name", "parameters"}]`` list, a single object, or the near-misses
      ``name`` / ``arguments``.
    - **JSON inside prose** (``embedded=True``): Aya Expanse often writes "I'll now call the
      tool: ```json [...]``` Please confirm", sometimes followed by a made-up "Tool result".
      The first JSON value using the requested ``tool_name`` key is taken as the call, and
      **everything else in the reply is dropped**, including any invented result after it.
      τ²-bench's rule is a message *or* a tool call, so the turn becomes the tool call.
    - A well-formed call to a tool that doesn't exist is kept: τ²-bench records it, and it is
      exactly the Tool hallucination we want to detect.
    """
    stripped = _FENCE.sub("", re.sub(r"^\s*Action:\s*", "", text.strip(), flags=re.IGNORECASE))
    if stripped.startswith(("[", "{")):
        first = next(_json_values(stripped), None)
        calls = _calls_from(first, id_prefix, strict=False)
        if calls:
            return calls, False
    for value in _json_values(text):
        calls = _calls_from(value, id_prefix, strict=True)
        if calls:
            return calls, True
    call = _functional_call(stripped)
    if call:
        return [{"id": f"{id_prefix}_0", "type": "function", "function": call}], False
    return None, False


_FUNCTIONAL = re.compile(r"^([A-Za-z_]\w*)\((.*)\)\s*$", re.DOTALL)


def _functional_call(text: str) -> dict | None:
    """``name({...})`` or ``name(key=value, ...)`` as the whole reply (seen from Tiny Aya, 3 Nov).

    JSON arguments are read directly; keyword arguments go through τ²-bench's own parser for
    this style (`tau2.utils.tools.parse_action_string`), when τ²-bench is installed.
    """
    m = _FUNCTIONAL.match(text.strip())
    if not m:
        return None
    name, inner = m.group(1), m.group(2).strip()
    if not inner:
        return {"name": name, "arguments": "{}"}
    if inner.startswith("{"):
        try:
            args = json.loads(inner)
        except ValueError:
            return None
        return {"name": name, "arguments": json.dumps(args)} if isinstance(args, dict) else None
    try:
        from tau2.utils.tools import parse_action_string

        msg = parse_action_string(f"{name}({inner})")
    except Exception:  # noqa: BLE001 - not a call after all
        return None
    calls = getattr(msg, "tool_calls", None) or []
    if len(calls) != 1:
        return None
    return {"name": calls[0].name, "arguments": json.dumps(calls[0].arguments)}


def parse_prompted(text: str, id_prefix: str = "call") -> list[dict] | None:
    """Tool calls from a prompted reply (see `find_prompted_calls`), or None."""
    return find_prompted_calls(text, id_prefix)[0]


# --- Client ----------------------------------------------------------------------------------


class RateLimiter:
    """Spaces calls at least 60/rpm seconds apart (thread-safe)."""

    def __init__(
        self,
        rpm: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
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
    embedded: int = 0  # prompted tool calls written inside prose (the prose was dropped)
    directly_answer: int = 0  # "directly-answer" pseudo-calls turned into a message
    empty: int = 0  # empty replies (re-asked once without stop sequences)
    cut_off: int = 0  # replies the API stopped at max_tokens (finish_reason "length")
    per_call: list[dict] = field(default_factory=list)
    prices: tuple[float, float] = (PRICE_IN, PRICE_OUT)  # USD per input / output token

    @property
    def cost(self) -> float:
        """Estimated USD at ``prices`` (default: Aya Expanse's listed PRICE_IN / PRICE_OUT)."""
        return self.prompt_tokens * self.prices[0] + self.completion_tokens * self.prices[1]

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "seconds": round(self.seconds, 2),
            "errors": self.errors,
            "unparsed": self.unparsed,
            "embedded": self.embedded,
            "directly_answer": self.directly_answer,
            "empty": self.empty,
            "cut_off": self.cut_off,
            "est_cost": round(self.cost, 4),
        }


_RETRYABLE = (
    "RateLimitError",
    "InternalServerError",
    "ServiceUnavailableError",
    "APIConnectionError",
    "Timeout",
    "APIError",
    "_NoResponse",
)


class _NoResponse(RuntimeError):
    """The API answered with nothing (seen once from Cohere, 3 Nov); retried."""


def _retryable(e: Exception) -> bool:
    """Rate limits, server errors and timeouts can succeed later; a bad request won't."""
    names = {c.__name__ for c in type(e).__mro__}
    return "BadRequestError" not in names and bool(names & set(_RETRYABLE))


class EmptyTeacherReply(RuntimeError):
    """The teacher returned nothing usable twice in a row for one turn."""


class ContextTooLong(RuntimeError):
    """The model's API refused the prompt as longer than its context window."""


# How each server words it: Cohere ("too many tokens: size limit exceeded"), vLLM and OpenAI
# ("This model's maximum context length is 32768 tokens"); litellm's own class for either
_TOO_LONG = ("too many tokens", "maximum context length")


def _too_long(e: Exception) -> bool:
    names = {c.__name__ for c in type(e).__mro__}
    return "ContextWindowExceededError" in names or any(s in str(e).lower() for s in _TOO_LONG)


class CohereClient:
    """Aya Expanse 32B (by default) as the agent: one call per agent turn."""

    def __init__(
        self,
        mode: str = "prompted",
        *,
        style: str = "cohere",
        model: str = TEACHER_MODEL,
        api_base: str = COHERE_COMPAT_BASE,
        api_key: str | None = None,
        rpm: float | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
        num_retries: int = 3,
        prices: tuple[float, float] | None = None,
        completion: Callable[..., Any] | None = None,
    ) -> None:
        if mode not in ("native", "prompted"):
            raise ValueError("mode must be 'native' or 'prompted'")
        if style not in ("agenttune", "cohere"):
            raise ValueError("style must be 'agenttune' or 'cohere'")
        self.mode, self.style, self.model, self.api_base = mode, style, model, api_base
        self.api_key = api_key or os.environ.get("COHERE_API_KEY")
        if not self.api_key and completion is None:
            raise RuntimeError("Set COHERE_API_KEY in .env (see docs/TEACHER_SETUP.md)")
        self.temperature, self.max_tokens, self.seed = temperature, max_tokens, seed
        self.num_retries = num_retries
        rpm = rpm if rpm is not None else float(os.environ.get("TEACHER_RPM") or 0)  # 0 = no limit
        self.limiter = RateLimiter(rpm)
        self._completion = completion
        self.usage = Usage(prices=prices) if prices is not None else Usage()
        # Prompted mode: each raw reply and how it was read, for reviewing the model's mistakes
        self.raw_turns: list[dict] = []

    def _call(self, messages: list[dict], tools: list[dict] | None, stop: list[str] | None = None) -> Any:
        if self._completion is None:
            import litellm

            litellm.suppress_debug_info = True  # ty: ignore[invalid-assignment]  # no "Get Help" banner per error
            self._completion = litellm.completion
        kwargs: dict[str, Any] = dict(
            model=f"openai/{self.model}",
            api_base=self.api_base,
            api_key=self.api_key,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_retries=0,  # we retry ourselves, and only errors that can succeed later
        )
        if tools:
            kwargs["tools"] = tools
        if stop:
            kwargs["stop"] = stop
        if self.seed is not None:
            kwargs["seed"] = self.seed
        start = time.monotonic()
        try:
            for attempt in range(self.num_retries + 1):
                self.limiter.wait()
                try:
                    resp = self._completion(**kwargs)
                    if resp is None or not getattr(resp, "choices", None):
                        raise _NoResponse("the API returned no choices")
                    break
                except Exception as e:
                    if _too_long(e):
                        raise ContextTooLong(str(e)) from e
                    if attempt == self.num_retries or not _retryable(e):
                        raise
                    time.sleep(min(2**attempt, 30))  # 1 s, 2 s, 4 s ...
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
        finish = getattr(resp.choices[0], "finish_reason", None)
        if finish == "length":  # stopped at max_tokens: the reply may be missing its end
            self.usage.cut_off += 1
        self.usage.per_call.append({"prompt_tokens": pt, "completion_tokens": ct, "finish_reason": finish})
        return resp

    def turn(self, messages: list[dict], tool_schemas: list[dict]) -> dict:
        """One agent turn for ``messages`` (from `Tau2Episode.prompt()`), as an assistant dict."""
        if self.mode == "native":
            msg = self._call(messages, tool_schemas).choices[0].message
            calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                }
                for tc in (getattr(msg, "tool_calls", None) or [])
            ]
            if calls:  # τ²-bench's rule: a tool call or a message, not both
                return {"role": "assistant", "content": None, "tool_calls": calls}
            return {"role": "assistant", "content": msg.content or ""}

        prompt = to_prompted(messages, tool_schemas, self.style)
        text = self._call(prompt, None, stop=STOP_SEQUENCES).choices[0].message.content or ""
        if not text.strip():
            # Usually a stop sequence at the very start (Aya began by echoing <results>). Ask
            # once more without stop sequences; anything results-like is dropped by the parser.
            self.usage.empty += 1
            nudge = [
                *prompt,
                {
                    "role": "user",
                    "content": "(Write your next turn now: an Action, or a message to the customer.)",
                },
            ]
            text = self._call(nudge, None).choices[0].message.content or ""
            if not text.strip():
                raise EmptyTeacherReply("The teacher returned an empty reply twice")
        calls, embedded = find_prompted_calls(text, id_prefix=f"t{self.usage.calls}")

        def note(kind: str) -> None:
            self.raw_turns.append({"turn": len(self.raw_turns) + 1, "kind": kind, "reply": text[:4000]})

        if calls and all(c["function"]["name"] in DIRECTLY_ANSWER for c in calls):
            note("directly_answer")
            return self._directly_answer(prompt, text)
        if calls:
            calls = [c for c in calls if c["function"]["name"] not in DIRECTLY_ANSWER]
            self.usage.embedded += embedded
            note("tool_call_in_prose" if embedded else "tool_call")
            return {"role": "assistant", "content": None, "tool_calls": calls}
        if text.strip().startswith(("[", "{")) or "tool_name" in text:
            self.usage.unparsed += 1  # looked like a call but wasn't valid: sent as a message
            note("unparsed_call")
        else:
            note("message")
        message = _strip_results(text)
        if not message:
            raise EmptyTeacherReply("The teacher's reply had no message and no tool call")
        return {"role": "assistant", "content": message}

    def _directly_answer(self, prompt: list[dict], text: str) -> dict:
        """The model chose `directly-answer`: its message to the customer, as a message turn.

        Uses any prose around the Action; if there is none, asks once more for the reply.
        """
        self.usage.directly_answer += 1
        prose = _strip_actions(text)
        if prose:
            return {"role": "assistant", "content": prose}
        follow = [
            *prompt,
            {"role": "assistant", "content": text.strip()},
            {"role": "user", "content": "(Now write your reply to the customer, as plain text.)"},
        ]
        reply = self._call(follow, None, stop=STOP_SEQUENCES).choices[0].message.content or ""
        return {"role": "assistant", "content": _strip_actions(reply) or reply}


TeacherClient = CohereClient  # the old name, still used by reflexive.arms

_ID_LIKE = re.compile(r"^(?=.*\d)[\w#@.\-]{4,}$")
# Dates and times are worked out from the conversation ("next Friday"), not looked up, so they
# aren't treated as invented IDs (3 Nov: every flag for the top teachers was a flight date).
_DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?$|^\d{1,2}:\d{2}$")


def unseen_arguments(messages: list[dict]) -> list[str]:
    """Agent tool-call arguments that look like IDs but never appeared earlier in the conversation.

    A diagnostic for teacher transcripts (SEL-9): a value like ``1012345678`` or
    ``credit_card_0000000`` that no customer message or tool result mentioned before the call is
    probably invented. Strings with a digit, 4+ characters and no spaces count as ID-like.
    """
    seen, unseen = "", []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        continue
                values = []
                for v in args.values():
                    values += v if isinstance(v, list) else [v]
                unseen += [
                    v
                    for v in values
                    if isinstance(v, str)
                    and _ID_LIKE.match(v)
                    and not _DATE_LIKE.match(v)
                    and v.lstrip("#") not in seen
                    and v not in unseen
                ]
        seen += " " + (m.get("content") or "")
    return unseen


def tool_errors(messages: list[dict]) -> dict:
    """Counts of two other tool hallucinations in a conversation (agent calls only).

    - ``unknown_tools``: calls to a tool the agent doesn't have (τ²-bench answers
      "Error: Tool '<name>' not found."), e.g. the customer-side telecom tools.
    - ``repeated_failures``: a call identical (same tool and arguments) to an earlier call that
      returned an error, i.e. retrying something that already failed (a loop).
    """
    results = {m.get("tool_call_id"): m.get("content") or "" for m in messages if m["role"] == "tool"}
    unknown = repeated = 0
    failed: set[str] = set()
    for m in messages:
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except ValueError:
                    args = {"_raw": args}
            key = f"{fn['name']}:{json.dumps(args, sort_keys=True)}"
            out = results.get(tc.get("id"), "")
            if out.startswith("Error: Tool '") and "not found" in out:
                unknown += 1
            if key in failed:
                repeated += 1
            if out.startswith("Error"):
                failed.add(key)
    return {"unknown_tools": unknown, "repeated_failures": repeated}


def _strip_results(text: str) -> str:
    """``text`` cut at the first thing that looks like an invented tool result."""
    cut = min((i for i in (text.find(s) for s in STOP_SEQUENCES) if i >= 0), default=len(text))
    return text[:cut].strip()


def _strip_actions(text: str) -> str:
    """``text`` without any 'Action:' JSON block (fenced or bare)."""
    text = re.sub(r"Action:\s*```(?:json)?.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"Action:\s*\[.*?\]\s*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()
