"""Heal GRPO data (SEL-75): the teacher's fixes at the student's own Tool / Arg / Loop failures.

The first GRPO run (SEL-21) trained on the teacher's clean conversations, and only on its tool-call
turns. This is the design in the README instead: **Detect & Classify → teacher fix → retrain**.

1. **Heal points** (`points_of`). A failed student conversation is classified by Chiara's
   classifier (`reflexive.hallucination`). For each failure type it contains, the **first finding
   of that type** is a point: the conversation as the student saw it, up to that turn. The first
   finding overall is usually an Arg failure, and loops come after it, so taking only the first
   finding would leave almost no Loop points. Two types on the same turn make one point with both
   labels. ``step_limit`` isn't a point: it isn't a turn the agent wrote.
2. **Fixes** (`scripts/heal/make_fixes.py`): the teacher rewrites that one turn
   (`reflexive.heal.PrefixTeacher`, `check_fix`), as Chiara's heal cycle does. A fix can be a tool
   call or a reply to the customer (asking for a missing ID is the usual Arg fix).
3. **GRPO prompts** (`build`): each fix becomes one prompt, the prefix in Aya's format, with the
   fix as the reference turn. **Anchor** prompts from the teacher's own conversations (replies as
   well as tool calls) are mixed in, so the model keeps doing the ordinary turns right; the first
   run's model, trained on tool calls only, stopped replying to customers.

Each row also carries what the loop reward needs: the calls already in the prompt, which of them
failed, and the agent's earlier replies (`history`), plus the failed turn itself.

Stdlib only.
"""

from __future__ import annotations

import collections
import json
import math
import random
import re
from collections.abc import Sequence
from typing import Any

from reflexive import grpo_data, splits
from reflexive.hallucination import Diagnosis, FailureType, Finding, classify_record

FORMAT = grpo_data.FORMAT  # the same prompt format as the first run, with more columns
TYPES = ("TOOL", "ARG", "LOOP")  # Chiara's failure types, in the classifier's order
NONE = "NONE"  # an anchor: no failure, the teacher's ordinary turn
FIX_GROUP = "heal"

Record = dict[str, Any]

# A tool result in Aya's format (`reflexive.teacher.COHERE_RESULT`); errors start "Error"
_RESULT = re.compile(r"<results>\s*Document: \d+\s*tool_name: [^\n]*\s*result: (.*?)\s*</results>", re.S)


class HealDataError(Exception):
    """A check failed: nothing is written. ``lines`` are printed with ⛔."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[-1])
        self.lines = lines


# --- Heal points -------------------------------------------------------------------------------


def points_of(record: Record) -> list[tuple[list[str], Finding]]:
    """``[(types, finding)]``: the first finding of each failure type, one point per turn."""
    diagnosis = classify_record(record)
    by_turn: dict[int, tuple[list[str], Finding]] = {}
    for t in TYPES:
        first = next(
            (f for f in diagnosis.findings if f.type.value == t and f.kind != "step_limit"),
            None,
        )
        if first is not None:
            by_turn.setdefault(first.turn, ([], first))[0].append(t)
    return [by_turn[turn] for turn in sorted(by_turn)]


def diagnosis_for(types: Sequence[str], finding: Finding) -> Diagnosis:
    """A diagnosis whose first finding is this point's, so `heal.fix_point` cuts the prefix there."""
    return Diagnosis(failure=FailureType(types[0]), findings=[finding])


def primary(labels: Sequence[str]) -> str:
    """The point's main failure type: Tool, then Arg, then Loop (the classifier's order)."""
    return next((t for t in TYPES if t in labels), NONE)


# --- What the prompt already holds ---------------------------------------------------------------


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().casefold())


def call_key(name: str, args: Any) -> str:
    return json.dumps([name, args], sort_keys=True, default=str)


def history(prompt: Sequence[Record]) -> Record:
    """The agent's earlier calls (with whether each failed) and replies in a prompt.

    ``calls``: ``[{"key", "failed"}]`` in order; a call failed if its result starts ``Error``.
    ``texts``: the agent's earlier replies to the customer, normalised."""
    calls: list[Record] = []
    texts: list[str] = []
    pending: list[str] = []
    for m in prompt:
        content = str(m.get("content") or "")
        if m.get("role") == "assistant":
            found = grpo_data.calls_of(content)
            if found:
                pending = [call_key(c["name"], c["arguments"]) for c in found]
            elif content.strip():
                texts.append(norm_text(content))
                pending = []
        elif m.get("role") == "user" and pending:
            results = _RESULT.findall(content)
            for i, key in enumerate(pending):
                result = results[i] if i < len(results) else ""
                calls.append({"key": key, "failed": result.lstrip().startswith("Error")})
            pending = []
    return {"calls": calls, "texts": texts}


def failed_of(example: Record) -> Record:
    """The student's failed turn at a heal point: its calls (as keys) and its text."""
    turn = (example.get("source") or {}).get("failed_turn") or {}
    keys = []
    for tc in turn.get("tool_calls") or []:
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = raw
        keys.append(call_key(tc["function"]["name"], args))
    return {"calls": keys, "text": norm_text(turn.get("content") or "") if not keys else ""}


def failed_text(failed: Record) -> str:
    """The failed turn as Aya wrote it (an ``Action:`` block, or the reply), for scoring it."""
    if failed.get("calls"):
        calls = [{"tool_name": n, "parameters": a} for n, a in (json.loads(k) for k in failed["calls"])]
        return f"Action: ```json\n{json.dumps(calls, indent=4)}\n```"
    return failed.get("text") or ""


# --- GRPO prompts -------------------------------------------------------------------------------


def turn_rows(example: Record, failure_type: str | None = None) -> list[Record]:
    """GRPO rows for an example's target turns (tool calls *and* replies), with the heal columns."""
    labels = list(example.get("labels") or [])
    is_fix = example.get("group") == FIX_GROUP
    ftype = failure_type or (primary(labels) if is_fix else NONE)
    failed = failed_of(example) if is_fix else {"calls": [], "text": ""}
    rows = []
    for r in grpo_data.turns_of(example, kinds=("call", "text")):
        rows.append(
            {
                **r,
                "failure_type": ftype,
                "failure_types": [t for t in TYPES if t in labels] if is_fix else [],
                "failed": failed,
                "history": history(r["prompt"]),
            }
        )
    return rows


def _problems(e: Record) -> list[str]:
    problems = grpo_data.problems_of(e)
    if e.get("group") == FIX_GROUP and primary(e.get("labels") or []) == NONE:
        problems.append(f"{e.get('id')}: a fix with no Tool / Arg / Loop label")
    return problems


def _summary(rows: list[Record]) -> Record:
    return {
        "prompts": len(rows),
        "tasks": len({(r["domain"], r["task_id"]) for r in rows}),
        "failure_type": dict(collections.Counter(r["failure_type"] for r in rows).most_common()),
        "kind": dict(collections.Counter(r["kind"] for r in rows).most_common()),
        "domain": dict(collections.Counter(r["domain"] for r in rows).most_common()),
        "type_by_kind": dict(
            collections.Counter(f"{r['failure_type']}/{r['kind']}" for r in rows).most_common()
        ),
    }


def build(
    fixes: Sequence[Record],
    golden: Sequence[Record],
    *,
    anchors_per_fix: float = 0.5,
    heldout_fraction: float = 0.1,
    seed: int = 0,
) -> tuple[list[Record], list[Record], Record]:
    """Training and held-out GRPO rows, plus a report. Raises `HealDataError`.

    ``fixes``: ``reagent.sft/v1`` heal fixes (`scripts/heal/make_fixes.py`); ``golden``: the
    teacher's conversations (``just train-data``), for anchors. A seeded share of the fixes'
    **tasks** is held out (with their anchors), never single turns: turns of one conversation
    share most of their prompt. ``anchors_per_fix`` anchors are drawn per fix row, on each side."""
    problems = [p for e in [*fixes, *golden] for p in _problems(e)]
    if problems:
        raise HealDataError(*problems[:10], f"{len(problems)} problem(s) in the inputs: nothing written")
    if not fixes:
        raise HealDataError("no fixes: run `just heal-fixes` first")
    for e in [*fixes, *golden]:
        splits.assert_not_test(e["domain"], str(e["task_id"]))
        if splits.split_of(e["domain"], str(e["task_id"])) != "train":
            raise HealDataError(f"{e['id']} isn't a train task: never train on dev or test")

    rng = random.Random(seed)
    tasks = sorted({(e["domain"], str(e["task_id"])) for e in fixes})
    rng.shuffle(tasks)
    held = set(tasks[: math.ceil(heldout_fraction * len(tasks))]) if heldout_fraction else set()

    def side(e: Record) -> str:
        return "heldout" if (e["domain"], str(e["task_id"])) in held else "train"

    out: dict[str, list[Record]] = {"train": [], "heldout": []}
    for e in fixes:
        out[side(e)] += turn_rows(e)
    for name in ("train", "heldout"):
        pool = [r for e in golden if side(e) == name for r in turn_rows(e, NONE)]
        n_fix = len(out[name])
        n = min(len(pool), round(anchors_per_fix * n_fix))
        out[name] += rng.sample(pool, n)
        rng.shuffle(out[name])

    ids = collections.Counter(r["id"] for rows in out.values() for r in rows)
    if dupes := [i for i, n in ids.items() if n > 1]:
        raise HealDataError(f"{len(dupes)} prompt id(s) twice, e.g. {dupes[0]}: are two fix files the same?")
    report = {
        "train": _summary(out["train"]),
        "heldout": _summary(out["heldout"]),
        "heldout_tasks": sorted(f"{d}/{t}" for d, t in held),
        "anchors_per_fix": anchors_per_fix,
        "seed": seed,
    }
    return out["train"], out["heldout"], report
