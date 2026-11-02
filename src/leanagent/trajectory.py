"""Parse an agent trajectory into tool calls, tool results and a final answer.

One parser shared by the reward functions (training) and the metrics (eval), so
what we optimise and what we report are measured the same way.

A trajectory ("completion" in TRL terms) is a list of chat turns. AgentTune's
rollout gives assistant turns with ``tool_calls`` and ``tool`` turns with results.
Cohere models without a tool-capable chat template use AgentTune's fallback: the
model replies with a JSON list ``[{"tool_name": ..., "parameters": {...}}]`` and
results come back as user turns starting ``"Tool result for <name>: ..."``.
Both shapes are handled here.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

UNANSWERABLE = "UNANSWERABLE"

ABSTAIN_PATTERNS = re.compile(
    r"\b(i (do not|don't|cannot|can't) (know|find|answer|determine)|"
    r"not (available|found|present|in the (data|database|context))|"
    r"no (information|data|record)s? (is |are )?(available|found)|"
    r"unanswerable|insufficient (information|data))\b",
    re.IGNORECASE,
)
ERROR_PATTERNS = re.compile(r"(^\s*error\b|\"error\"|'error'|traceback|exception:)", re.IGNORECASE)
TOOL_RESULT_PREFIX = re.compile(r"^\s*Tool result for (?P<name>[\w\-.]+):\s*(?P<content>.*)", re.S)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    well_formed: bool = True  # arguments parsed as a JSON object

    def key(self) -> str:
        """Identity used to spot redundant (repeated identical) calls."""
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, default=str)}"


@dataclass
class Trajectory:
    calls: list[ToolCall] = field(default_factory=list)
    results: list[str] = field(default_factory=list)
    final_answer: str = ""

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_redundant(self) -> int:
        seen: set[str] = set()
        dup = 0
        for c in self.calls:
            k = c.key()
            dup += k in seen
            seen.add(k)
        return dup

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.results if ERROR_PATTERNS.search(r or ""))

    @property
    def n_malformed(self) -> int:
        return sum(1 for c in self.calls if not c.well_formed)

    @property
    def successful_results(self) -> list[str]:
        return [r for r in self.results if r and not ERROR_PATTERNS.search(r)]

    @property
    def abstained(self) -> bool:
        return bool(ABSTAIN_PATTERNS.search(self.final_answer or ""))


def _parse_args(raw: Any) -> tuple[dict[str, Any], bool]:
    if isinstance(raw, dict):
        return raw, True
    if raw in (None, ""):
        return {}, True
    try:
        val = json.loads(raw)
        return (val, True) if isinstance(val, dict) else ({"_raw": raw}, False)
    except (TypeError, ValueError):
        return {"_raw": str(raw)}, False


def _calls_from_tool_calls(tool_calls: list[Any]) -> list[ToolCall]:
    out = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", tc)
        name = fn.get("name") or fn.get("tool_name") or ""
        args, ok = _parse_args(fn.get("arguments", fn.get("parameters")))
        out.append(ToolCall(name=name, arguments=args, well_formed=ok and bool(name)))
    return out


_JSON_LIST = re.compile(r"\[\s*\{.*\}\s*\]", re.S)
_COHERE_ACTION = re.compile(r"<\|START_ACTION\|>(.*?)<\|END_ACTION\|>", re.S)


def parse_text_tool_calls(text: str) -> list[ToolCall] | None:
    """Tool calls written as text (AgentTune fallback JSON list or Cohere action block).

    Returns None when the text is not a tool-call turn (i.e. it is an answer).
    """
    if not text:
        return None
    m = _COHERE_ACTION.search(text) or _JSON_LIST.search(text)
    if not m:
        return None
    blob = m.group(1) if m.re is _COHERE_ACTION else m.group(0)
    try:
        items = json.loads(blob)
    except ValueError:
        return [ToolCall(name="", arguments={"_raw": blob}, well_formed=False)]
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return None
    if not any("tool_name" in i or "name" in i for i in items):
        return None
    return _calls_from_tool_calls(items)


def to_turns(completion: Any) -> list[dict[str, Any]]:
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    if isinstance(completion, dict):
        return [completion]
    return list(completion or [])


def parse(completion: Any) -> Trajectory:
    traj = Trajectory()
    last_answer = ""
    for turn in to_turns(completion):
        role = turn.get("role")
        content = turn.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content, default=str)
        if role == "assistant":
            if turn.get("tool_calls"):
                traj.calls.extend(_calls_from_tool_calls(turn["tool_calls"]))
                continue
            text_calls = parse_text_tool_calls(content)
            if text_calls is not None:
                traj.calls.extend(text_calls)
                continue
            if content.strip():
                last_answer = content
        elif role == "tool":
            traj.results.append(content)
        elif role == "user":
            m = TOOL_RESULT_PREFIX.match(content)
            if m:
                traj.results.append(m.group("content"))
    traj.final_answer = last_answer.strip()
    return traj


_STARRED = re.compile(r"\*{1,2}([^*\n]+?)\*{1,2}")


def extract_answer(final_answer: str) -> str:
    """The short answer: last ``*starred*`` span if present, else the whole text."""
    hits = _STARRED.findall(final_answer or "")
    return (hits[-1] if hits else final_answer or "").strip()


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", (s or "").lower()).strip()


def is_correct(final_answer: str, gold: str | None) -> bool:
    if gold is None:
        return False
    if str(gold).upper() == UNANSWERABLE:
        return bool(ABSTAIN_PATTERNS.search(final_answer or ""))
    pred, ref = normalize(extract_answer(final_answer)), normalize(str(gold))
    return bool(ref) and (pred == ref or pred.startswith(ref + " "))


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def unsupported_numbers(final_answer: str, results: list[str]) -> list[str]:
    """Numbers stated in the answer that appear in no tool result (a hallucination signal)."""
    evidence = " ".join(results)
    return [n for n in _NUMBER.findall(final_answer or "") if n not in evidence]
