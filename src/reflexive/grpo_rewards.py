"""The reward for single-turn GRPO (SEL-21): one sampled agent turn against the teacher's.

Each GRPO prompt (`reflexive.grpo_data`) is a conversation up to an agent turn; the model
samples that turn several times, and `score` rates each sample against the teacher's verified
turn there. No customer, no environment: everything is read from the prompt and the reference.

**Components** (each 0–1; the reward is their weighted sum, weights in the training config):

* ``tool_match``: the sample calls the same tools as the teacher (as a multiset; partial credit
  by overlap). For a text reference: the sample is text too.
* ``args_match``: each teacher call's arguments against the paired sample call (the share of
  the teacher's arguments, after normalising case, spaces and number formats).
* ``valid_call``: every call names one of the agent's tools (not the customer's), with no
  unknown or missing parameter and the right types. A text sample: it says something
  (``MIN_TEXT``).
* ``grounded``: no ID-like argument is invented or a placeholder; each appeared in the prompt
  (`reflexive.hallucination`'s checks: what the classifier flags as Arg hallucinations).
* ``format``: the tool call is the whole reply (``Action: ```json [...]``` ``), not JSON inside
  prose, which τ²-bench reads as a call but drops the rest of.
* ``no_give_up``: the sample doesn't stop, hand off to a human or say a few words, *unless the
  teacher did the same there*: the 8 verified hand-offs in the data are right, not giving up.

Defaults weight getting the action right (tool and arguments) over the checks any reply can
pass; see ``DEFAULT_WEIGHTS``. The teacher's own turns score the maximum on all of them
(checked on the real data with ``just grpo-reward-check``).

**Dataset columns** the reward reads (TRL passes every column as a keyword argument):
``target_calls`` (JSON text: Arrow can't hold argument dicts of different shapes in one column),
``target`` (the teacher's turn), ``domain``, ``id``; and TRL's own ``prompts``.
"""

from __future__ import annotations

import collections
import functools
import json
import re
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from reflexive.hallucination import _arg_findings, _Context, domain_tools
from reflexive.heal import HANDOFF, MIN_TEXT, STOP
from reflexive.teacher import find_prompted_calls

COMPONENTS = ("tool_match", "args_match", "valid_call", "grounded", "format", "no_give_up")
DEFAULT_WEIGHTS = {
    "tool_match": 1.0,
    "args_match": 1.0,
    "valid_call": 0.5,
    "grounded": 0.5,
    "format": 0.2,
    "no_give_up": 0.5,
}
_VALIDITY = {"unknown_param", "missing_param", "wrong_type"}
_GROUNDING = {"placeholder", "ungrounded"}

Call = dict[str, Any]


def completion_text(completion: Any) -> str:
    """The sampled turn's text: TRL gives a string, or a list of messages for a chat dataset."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for m in reversed(completion):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content") or "")
        return ""
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    return str(completion or "")


def parse_calls(text: str) -> tuple[list[Call], bool]:
    """``([{"name", "arguments"}], embedded)``: the turn's calls, and whether they sat inside prose."""
    raw, embedded = find_prompted_calls(text)
    calls = []
    for c in raw or []:
        a = c["function"].get("arguments") or "{}"
        try:
            args = json.loads(a) if isinstance(a, str) else a
        except json.JSONDecodeError:
            args = {}
        calls.append({"name": c["function"]["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls, embedded


def _norm(v: Any) -> Any:
    """A value as compared: case, spaces and number formats don't count; lists in any order."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        s = re.sub(r"\s+", " ", v.strip().casefold())
        if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
            return float(s)
        return s
    if isinstance(v, list):
        return tuple(sorted(json.dumps(_norm(x), sort_keys=True, default=str) for x in v))
    if isinstance(v, dict):
        return json.dumps({k: _norm(x) for k, x in v.items()}, sort_keys=True, default=str)
    return v


def args_score(target: dict, sample: dict) -> float:
    """Share of the teacher's arguments the sample gives with the same value (1.0 if it has none)."""
    if not target:
        return 1.0 if not sample else 0.5  # the teacher passed nothing; extra arguments half right
    hits = sum(1 for k, v in target.items() if k in sample and _norm(sample[k]) == _norm(v))
    return hits / len(target)


def _pair(target: Sequence[Call], sample: Sequence[Call]) -> list[tuple[Call, Call | None]]:
    """Each teacher call with a sample call of the same tool (or None), best-matching pairs first.

    Greedy over all pairs, not teacher call by teacher call: with two teacher calls to one tool
    and one sample call, the sample goes to the call it matches, whichever comes first.
    """
    candidates = sorted(
        (
            (args_score(t["arguments"], s["arguments"]), -ti, -si)
            for ti, t in enumerate(target)
            for si, s in enumerate(sample)
            if s["name"] == t["name"]
        ),
        reverse=True,
    )
    match: dict[int, int] = {}
    for _, nti, nsi in candidates:
        if -nti not in match and -nsi not in match.values():
            match[-nti] = -nsi
    return [(t, sample[match[ti]] if ti in match else None) for ti, t in enumerate(target)]


def _overlap(target: Sequence[Call], sample: Sequence[Call]) -> float:
    a = collections.Counter(c["name"] for c in target)
    b = collections.Counter(c["name"] for c in sample)
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 1.0


def _gives_up(text: str, calls: Sequence[Call]) -> bool:
    if STOP in text or any(c["name"] == HANDOFF for c in calls):
        return True
    return not calls and len(text.strip()) < MIN_TEXT


@functools.lru_cache(maxsize=1024)  # ~30 KB each: every prompt of a build
def _context(contents: tuple[str, ...]) -> _Context:
    """What the agent had been told before the turn. Cached: a group's samples share one prompt,
    and building it (the ~7K-token system prompt, folded and searched) is most of `score`'s time.
    Read only: never add to the returned context."""
    ctx = _Context()
    for c in contents:
        ctx.add(c)
    return ctx


def score(
    text: str, target_calls: Sequence[Call], domain: str, prompt: Sequence[dict], target: str = ""
) -> dict:
    """Score one sampled turn. Returns every component (0–1) plus what it found, for the log."""
    calls, embedded = parse_calls(text)
    tools, customer = domain_tools(domain)
    schemas = {t["function"]["name"]: t["function"] for t in tools}
    ctx = _context(tuple(str(m.get("content") or "") for m in prompt))

    unknown_tools = [c["name"] for c in calls if c["name"] not in schemas]
    kinds = set()
    for i, c in enumerate(calls):
        if c["name"] in schemas:
            kinds |= {f.kind for f in _arg_findings(0, i, c["name"], c["arguments"], schemas[c["name"]], ctx)}

    if target_calls:
        tool_match = _overlap(target_calls, calls)
        pairs = _pair(target_calls, calls)
        args_match = sum(args_score(t["arguments"], s["arguments"]) if s else 0.0 for t, s in pairs) / len(
            pairs
        )
    else:  # the teacher talked to the customer here: a sample that calls a tool is the wrong move
        tool_match = args_match = 0.0 if calls else 1.0
    valid = (not unknown_tools and not kinds & _VALIDITY) if calls else len(text.strip()) >= MIN_TEXT
    teacher_gave_up = _gives_up(target, target_calls) if target else False
    return {
        "tool_match": tool_match,
        "args_match": args_match,
        "valid_call": float(valid),
        "grounded": float(not kinds & _GROUNDING),
        "format": float(not embedded) if calls else 1.0,
        "no_give_up": float(teacher_gave_up or not _gives_up(text, calls)),
        "n_calls": len(calls),
        "tools": [c["name"] for c in calls],
        "unknown_tools": unknown_tools,
        "customer_tools": [n for n in unknown_tools if n in set(customer)],
        "findings": sorted(kinds),
    }


def agent_tool_names(domain: str) -> list[str]:
    """The agent's tool names for a domain, in the schema file's order."""
    return [t["function"]["name"] for t in domain_tools(domain)[0]]


def total(s: dict, weights: dict[str, float]) -> float:
    return sum(w * s[k] for k, w in weights.items())


def _as_calls(v: Any) -> list[Call]:
    return json.loads(v) if isinstance(v, str) else list(v or [])


def make_reward_fn(
    weights: dict[str, float] | None = None, log_path: str | Path | None = None
) -> Callable[..., list[float]]:
    """A TRL / AgentTune reward function: the weighted sum of `score`'s components.

    Every scored sample is appended to ``log_path`` as one JSON line, with each component (open it
    with ``pd.read_json(path, lines=True)``): look for groups whose samples all score the same,
    which teach GRPO nothing. Under TRL, each component's batch mean is also logged as a metric
    (``reward_parts/<component>`` in W&B), through the ``log_metric`` it passes.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    if bad := sorted(set(w) - set(COMPONENTS)):
        raise ValueError(f"unknown reward component(s) {bad}: expected some of {COMPONENTS}")
    log = Path(log_path) if log_path else None
    if log and log.exists() and log.stat().st_size:
        log.rename(log.with_name(f"{log.stem}.stopped-{int(time.time())}{log.suffix}"))  # keep runs apart
    lock = threading.Lock()
    state = {"calls": 0}

    def single_turn_reward(
        completions: list[Any],
        prompts: list[Any] | None = None,
        target_calls: list[Any] | None = None,
        target: list[str] | None = None,
        domain: list[str] | None = None,
        id: list[str] | None = None,  # noqa: A002 - the dataset column's name
        **kwargs: Any,
    ) -> list[float]:
        if target_calls is None or domain is None:
            raise ValueError("the training dataset needs 'target_calls' and 'domain' columns")
        n = len(completions)
        state["calls"] += 1
        step = getattr(kwargs.get("trainer_state"), "global_step", None)
        rewards, rows = [], []
        for i in range(n):
            prompt = prompts[i] if prompts is not None and isinstance(prompts[i], list) else []
            text = completion_text(completions[i])
            s = score(text, _as_calls(target_calls[i]), domain[i], prompt, target[i] if target else "")
            r = total(s, w)
            rewards.append(r)
            rows.append(
                {
                    "step": step,
                    "call": state["calls"],
                    "id": id[i] if id else None,
                    "reward": r,
                    **s,
                    "completion": text[:2000],
                }
            )
        if log:
            with lock, log.open("a") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        if callable(log_metric := kwargs.get("log_metric")) and rows:  # TRL ≥ 1.x: one chart each
            for k in COMPONENTS:
                log_metric(f"reward_parts/{k}", sum(r[k] for r in rows) / len(rows))
        return rewards

    return single_turn_reward
