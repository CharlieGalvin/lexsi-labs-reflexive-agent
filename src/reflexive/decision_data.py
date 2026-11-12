"""Decision GRPO (arm H2): which state changes are still expected at a turn, and the reward for them.

SFT v2's failures on dev and validation are mostly **wrong decisions about what to change**
(39 of 55): a change nobody asked for, the right tool with wrong arguments, or an expected change
never made. The hallucination classifier sees almost none of them, so H (SEL-75), which learned
from the classifier's Tool / Arg / Loop points, couldn't learn them. H2 adds one reward that
checks the decision against the task's own ground truth: τ²-bench's expected actions.

* `decision_column`: for a GRPO prompt, the agent's expected state-changing actions not yet done
  in the prompt (``pending``), and the domain's state-changing tools (``writes``). Built once,
  with τ²-bench, when the data is written; the reward then needs no τ²-bench.
* `decision_score`: one sampled turn against that column (0–1, see its docstring).
* `decision_rows`: the teacher's turns at its decision points, as extra GRPO prompts.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from functools import cache
from typing import Any

from reflexive import grpo_data
from reflexive.heal_data import _RESULT

Record = dict[str, Any]
DECISION = "DECISION"  # the failure_type of a decision prompt
HOLD_OFF = 0.4  # no wrong change, but not the change the teacher made here (e.g. asked first)
EARLY = 0.5  # a pending change made where the teacher didn't make one yet (e.g. before confirming)


@cache
def _domain(domain: str) -> tuple[Any, tuple[str, ...]]:
    from loguru import logger
    from tau2.registry import registry

    from reflexive.outcome import scorer

    logger.remove()
    sc = scorer(domain)
    tools = registry.get_env_constructor(domain)(**(sc.env_kwargs or {})).tools
    writes = tuple(sorted(n for n in tools.get_tools() if tools.tool_mutates_state(n)))
    return sc, writes


def write_tools(domain: str) -> list[str]:
    """The agent's state-changing tools in a domain (τ²-bench's own list)."""
    return list(_domain(domain)[1])


def expected_writes(domain: str, task_id: str) -> list[list[Any]]:
    """The task's expected state-changing actions on the agent's side: ``[name, args, compare_args]``."""
    sc, writes = _domain(domain)
    actions = sc.task(str(task_id)).evaluation_criteria.actions or []
    return [
        [a.name, a.arguments or {}, a.compare_args]
        for a in actions
        if a.requestor != "user" and a.name in writes
    ]


def same_call(expected: Sequence[Any], name: str, args: Record) -> bool:
    """τ²-bench's rule (``Action.compare_with_tool_call``), as `failure_diff.same_call`: same tool,
    and the arguments agree on ``compare_args`` if the action names them, else on the keys sent."""
    if expected[0] != name:
        return False
    keys = expected[2] if len(expected) > 2 and expected[2] is not None else args.keys()
    return {k: v for k, v in args.items() if k in keys} == {
        k: v for k, v in (expected[1] or {}).items() if k in keys
    }


def done_calls(prompt: Sequence[Record]) -> list[tuple[str, Record]]:
    """The agent's calls in a prompt that went through (their result doesn't start ``Error``)."""
    done: list[tuple[str, Record]] = []
    waiting: list[tuple[str, Record]] = []
    for m in prompt:
        content = str(m.get("content") or "")
        if m.get("role") == "assistant":
            found = grpo_data.calls_of(content)
            waiting = [(c["name"], c["arguments"] if isinstance(c["arguments"], dict) else {}) for c in found]
        elif m.get("role") == "user" and waiting:
            results = _RESULT.findall(content)
            for i, call in enumerate(waiting):
                result = results[i] if i < len(results) else ""
                if not result.lstrip().startswith("Error"):
                    done.append(call)
            waiting = []
    return done


def pending(expected: Sequence[Sequence[Any]], prompt: Sequence[Record]) -> list[list[Any]]:
    """The expected changes not yet made in the prompt (each made call uses up one expected one)."""
    left = [list(e) for e in expected]
    for name, args in done_calls(prompt):
        hit = next((e for e in left if same_call(e, name, args)), None)
        if hit is not None:
            left.remove(hit)
    return left


def decision_column(domain: str, task_id: str, prompt: Sequence[Record]) -> Record:
    """The ``decision`` column of a GRPO row."""
    return {"pending": pending(expected_writes(domain, task_id), prompt), "writes": write_tools(domain)}


def decision_score(
    calls: Sequence[Record], target_calls: Sequence[Record], column: Record | None
) -> tuple[float, list[str]]:
    """How right the sampled turn's state changes are, and what was found.

    * A state-changing call that matches no pending expected change (a change nobody asked for,
      or the right tool with wrong arguments): **0**.
    * Where the teacher made a pending change: **1** if the sample makes it too, else
      `HOLD_OFF` (it changed nothing wrongly, but didn't do what was needed).
    * Where the teacher made no change: **1**; `EARLY` if the sample makes a pending change
      anyway (τ²-bench's policies want the customer's confirmation first).

    No column (H's rows): 1.0, so H's rewards are unchanged."""
    if not column:
        return 1.0, []
    writes = set(column.get("writes") or [])
    todo = [list(e) for e in column.get("pending") or []]
    found: list[str] = []
    made = []
    for c in calls:
        if c["name"] not in writes:
            continue
        args = c["arguments"] if isinstance(c["arguments"], dict) else {}
        hit = next((e for e in todo if same_call(e, c["name"], args)), None)
        if hit is None:
            same_tool = any(e[0] == c["name"] for e in todo)
            found.append(f"{c['name']}: {'wrong arguments' if same_tool else 'not expected'}")
        else:
            todo.remove(hit)
            made.append(hit)
    if found:
        return 0.0, found
    teacher_writes = [c for c in target_calls if c["name"] in writes]
    if teacher_writes:
        return (1.0, ["made"]) if made else (HOLD_OFF, ["held off"])
    return (EARLY, ["early"]) if made else (1.0, [])


def decision_rows(
    golden: Sequence[Record],
    *,
    exclude_tasks: set[tuple[str, str]],
    n: int,
    seed: int = 0,
) -> list[Record]:
    """Up to ``n`` GRPO rows at the teacher's decision points: each turn where it makes a state
    change, and the turn just before (often the confirmation). Tasks in ``exclude_tasks`` (held
    out, validation) are skipped. Columns as `reflexive.heal_data.turn_rows`, label `DECISION`."""
    from reflexive.heal_data import turn_rows

    picked: list[Record] = []
    for e in golden:
        if (e["domain"], str(e["task_id"])) in exclude_tasks:
            continue
        writes = set(write_tools(e["domain"]))
        rows = sorted(turn_rows(e, DECISION), key=lambda r: r["turn"])
        keep: set[int] = set()
        for j, r in enumerate(rows):
            if any(c["name"] in writes for c in r["target_calls"]):
                keep.add(j)
                if j:
                    keep.add(j - 1)
        picked += [rows[j] for j in sorted(keep)]
    rng = random.Random(seed)
    rng.shuffle(picked)
    return picked[:n]


def as_json(column: Record) -> str:
    return json.dumps(column, ensure_ascii=False)
