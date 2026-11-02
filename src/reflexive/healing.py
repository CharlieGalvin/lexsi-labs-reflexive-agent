"""Detect & Classify: the first stage of the Reflexive self-healing loop.

``classify_hallucination`` is a deterministic, groundedness-based classifier that labels a
failed trajectory as one of:

* ``TOOL`` — called a tool that does not exist (ToolNotFound) or an obviously wrong tool
* ``ARG``  — right tool, but hallucinated arguments: unknown/missing parameters, malformed
             JSON, or identifier-like values (ids, emails, dates, numbers) that appear nowhere
             in the task or in earlier tool results
* ``LOOP`` — repeats the same call, or runs out of steps without answering
* ``NONE`` — no structural failure found (the answer may still be wrong: see ``outcome``)

It is cheap and reproducible, so it can run on every EventLog. AgentTune's LLM-based
``agenttune.decide.closed_loop.FailureClassifier`` (root causes: wrong_tool, wrong_routing,
incomplete_reasoning, hallucinated_output, loop_collapse) can be layered on top for the
cases this one marks ``NONE``. ``TO_AGENTTUNE`` maps our labels onto its taxonomy.
"""

from __future__ import annotations

import inspect
import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from . import trajectory as T


class FailureType(StrEnum):
    TOOL = "TOOL"
    ARG = "ARG"
    LOOP = "LOOP"
    NONE = "NONE"


TO_AGENTTUNE = {FailureType.TOOL: "wrong_tool", FailureType.ARG: "hallucinated_output",
                FailureType.LOOP: "loop_collapse"}


@dataclass
class Diagnosis:
    failure: FailureType
    step: int | None = None            # index of the first offending tool call
    reasons: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"failure": self.failure.value, "step": self.step,
                "reasons": self.reasons, "evidence": self.evidence}


@dataclass
class ToolSpec:
    name: str
    params: set[str]
    required: set[str]

    @classmethod
    def from_callable(cls, fn: Callable) -> ToolSpec:
        sig = inspect.signature(fn)
        params = {p for p in sig.parameters if p not in ("self", "kwargs", "args")}
        required = {n for n, p in sig.parameters.items()
                    if n in params and p.default is inspect.Parameter.empty}
        return cls(fn.__name__, params, required)


# Values that a model should copy from context, never invent.
_ID_LIKE = re.compile(
    r"^(?:[\w.+-]+@[\w-]+\.[\w.]+"            # email
    r"|\d{4}-\d{2}-\d{2}"                     # ISO date
    r"|[A-Za-z]{0,6}[-_]?\d{2,}[\w-]*"        # ids like u_1042, ORD-77, 98231
    r"|-?\d+(?:\.\d+)?)$"                     # bare numbers
)
_IGNORE_SMALL = {"0", "1", "2", "3", "5", "10", "30", "100"}  # limits, windows, top-k


def _scalars(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [s for v in value.values() for s in _scalars(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _scalars(v)]
    if isinstance(value, bool) or value is None:
        return []
    return [str(value)]


def _context_text(turns: list[dict[str, Any]]) -> str:
    """Task text from user/system turns, excluding tool results (those are added in order)."""
    return " ".join(str(t.get("content") or "") for t in turns
                    if t.get("role") in ("user", "system")
                    and not T.TOOL_RESULT_PREFIX.match(str(t.get("content") or "")))


def ungrounded_values(args: dict[str, Any], context: str) -> list[str]:
    """Identifier-like argument values that never appeared in the task or prior results.

    SQL / free-text arguments are skipped as a whole; only short, id-like scalars are checked.
    """
    bad = []
    for v in _scalars(args):
        v = v.strip()
        if len(v) > 64 or " " in v or v in _IGNORE_SMALL:
            continue
        if _ID_LIKE.match(v) and v not in context:
            bad.append(v)
    return bad


def classify_hallucination(completion: Any, tools: list[Callable] | list[ToolSpec],
                           prompt: list[dict[str, Any]] | None = None,
                           max_steps: int | None = None, loop_threshold: int = 3) -> Diagnosis:
    turns = T.to_turns(completion)
    specs = {s.name: s for s in (t if isinstance(t, ToolSpec) else ToolSpec.from_callable(t)
                                 for t in tools)}
    traj = T.parse(turns)
    context = _context_text(prompt or []) + " " + _context_text(turns)

    # TOOL: unknown tool, or the environment said so
    for i, call in enumerate(traj.calls):
        if call.name and call.name not in specs:
            return Diagnosis(FailureType.TOOL, i, [f"tool {call.name!r} does not exist"],
                             {"known_tools": sorted(specs)})
    for i, r in enumerate(traj.results):
        if re.search(r"unknown tool|tool ?not ?found|no such tool", r or "", re.I):
            return Diagnosis(FailureType.TOOL, i, ["environment reported an unknown tool"])

    # ARG: schema violations or invented identifiers (grounded only in what came *before*)
    seen_results: list[str] = []
    for i, call in enumerate(traj.calls):
        spec = specs.get(call.name)
        reasons = []
        if not call.well_formed:
            reasons.append("malformed / non-JSON arguments")
        elif spec:
            extra = set(call.arguments) - spec.params
            missing = spec.required - set(call.arguments)
            if extra:
                reasons.append(f"unknown parameter(s) {sorted(extra)}")
            if missing:
                reasons.append(f"missing required parameter(s) {sorted(missing)}")
            invented = ungrounded_values(call.arguments, context_before(context, seen_results))
            if invented:
                reasons.append(f"ungrounded value(s) {invented}")
        if reasons:
            return Diagnosis(FailureType.ARG, i, reasons, {"call": call.name, "arguments": call.arguments})
        if i < len(traj.results):
            seen_results.append(traj.results[i])

    # LOOP: repeated identical calls, or ran out of budget without answering
    counts = Counter(c.key() for c in traj.calls)
    key, n = counts.most_common(1)[0] if counts else ("", 0)
    if n >= loop_threshold:
        first = next(i for i, c in enumerate(traj.calls) if c.key() == key)
        return Diagnosis(FailureType.LOOP, first, [f"identical call repeated {n}x"], {"call": key})
    if not traj.final_answer and traj.calls and (max_steps is None or traj.n_calls >= max_steps):
        return Diagnosis(FailureType.LOOP, traj.n_calls - 1, ["step budget exhausted without an answer"])

    return Diagnosis(FailureType.NONE)


def context_before(base_context: str, prior_results: list[str]) -> str:
    return base_context + " " + " ".join(prior_results)


def to_agenttune_failure(diag: Diagnosis, trajectory_id: str, context: dict[str, Any]) -> Any:
    """Wrap a Diagnosis as AgentTune's closed-loop ``Failure`` so its TrainingExampleGenerator,
    RetrainingTrigger and DeploymentGate can take over (imported lazily)."""
    from agenttune.decide.closed_loop.contracts import Failure

    return Failure(trajectory_id=trajectory_id, failure_type=diag.failure.value,
                   failed_stage_name=f"tool_call_{diag.step}",
                   context={**context, "diagnosis": diag.to_dict()},
                   error_message="; ".join(diag.reasons) or None)


def dumps(diag: Diagnosis) -> str:
    return json.dumps(diag.to_dict())
