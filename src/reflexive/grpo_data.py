"""Single-turn GRPO prompts from the verified teacher fixes (SEL-21).

GRPO here doesn't play whole τ²-bench conversations: that would need the simulated customer
(GPT-4.1) for every sample and the 32B student generating inside the training loop. Instead,
every agent turn of a verified training conversation (``reagent.sft/v1``, `reflexive.train_data`
or `reflexive.heal.to_sft_example`) becomes one GRPO prompt:

* ``prompt``: the conversation up to that turn (system, then alternating turns, ending on the
  customer or a ``<results>`` turn), exactly as Aya reads it at run time;
* ``target``: the teacher's turn there, the reference the reward compares a sample against;
* ``target_calls``: that turn's tool calls, parsed (``[]`` for a text turn).

**Which turns.** ``kinds=("call",)`` (default) keeps the tool-call turns only: they carry a clear
reward (right tool, right arguments). Text turns (``"text"``) are mostly the agent talking to
the customer, where a reward against one reference reply says little; add them with ``kinds``.

**Held out by conversation, never by turn.** The turns of one conversation share most of their
prompt, so a train / held-out split by turn would leak. `build` takes the two sets of
conversations separately (default in the script: the healable fixes to train on, the
``solved_by_aya`` ones held out, as the SFT run holds them out) and refuses a task in both.

Stdlib only: no τ²-bench, no GPU.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable, Sequence
from typing import Any

from reflexive import splits
from reflexive.teacher import parse_prompted

FORMAT = "reagent.grpo-turn/v1"
SOURCE_FORMAT = "reagent.sft/v1"
KINDS = ("call", "text")

Record = dict[str, Any]


class GrpoDataError(Exception):
    """A check failed: nothing is written. ``lines`` are printed with ⛔."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[-1])
        self.lines = lines


def calls_of(text: str) -> list[dict[str, Any]]:
    """The tool calls in an agent turn as ``[{"name", "arguments"}]`` (``[]`` for a text turn)."""
    calls = parse_prompted(text) or []
    out = []
    for c in calls:
        raw = c["function"].get("arguments") or "{}"
        args = json.loads(raw) if isinstance(raw, str) else raw
        out.append({"name": c["function"]["name"], "arguments": args})
    return out


def problems_of(example: Record) -> list[str]:
    """Why a conversation can't be cut into GRPO prompts (empty if it can)."""
    eid = example.get("id", "?")
    if example.get("format") != SOURCE_FORMAT:
        return [f"{eid}: format {example.get('format')!r}, expected {SOURCE_FORMAT!r}"]
    msgs, problems = example.get("messages") or [], []
    if not msgs or msgs[0].get("role") != "system":
        problems.append(f"{eid}: doesn't start with the system prompt")
    domain, task = str(example.get("domain")), str(example.get("task_id"))
    try:
        split = splits.split_of(domain, task)
    except (KeyError, ValueError) as e:
        return [*problems, f"{eid}: unknown task {domain}/{task} ({e})"]
    if split != "train":
        problems.append(f"{eid}: {domain}/{task} is a {split} task: only train tasks are trained on")
    for i in example.get("train_on") or []:
        if not 0 < i < len(msgs):
            problems.append(f"{eid}: train_on index {i} is outside the conversation")
        elif msgs[i].get("role") != "assistant":
            problems.append(f"{eid}: train_on index {i} isn't an agent turn")
        elif msgs[i - 1].get("role") == "assistant":
            problems.append(f"{eid}: the prompt for turn {i} ends on an agent turn")
    if not example.get("train_on"):
        problems.append(f"{eid}: no agent turns to learn from (train_on is empty)")
    return problems


def turns_of(example: Record, kinds: Sequence[str] = ("call",)) -> list[Record]:
    """One GRPO prompt per agent turn of ``example`` whose kind is in ``kinds``."""
    msgs = example["messages"]
    out = []
    for i in example["train_on"]:
        target = msgs[i]["content"]
        calls = calls_of(target)
        kind = "call" if calls else "text"
        if kind not in kinds:
            continue
        out.append(
            {
                "id": f"{example['id']}#t{i}",
                "format": FORMAT,
                "domain": example["domain"],
                "task_id": str(example["task_id"]),
                "group": example.get("group"),
                "labels": example.get("labels", []),
                "turn": i,
                "kind": kind,
                "prompt": [{"role": m["role"], "content": m["content"]} for m in msgs[:i]],
                "target": target,
                "target_calls": calls,
                "source": {"example_id": example["id"], "format": example["format"]},
            }
        )
    return out


def approx_tokens(messages: Iterable[Record]) -> int:
    """~4 characters a token: enough to size a run; the trainer counts exactly."""
    return sum(len(m["content"]) for m in messages) // 4


def _summary(rows: list[Record]) -> Record:
    lengths = sorted(approx_tokens(r["prompt"]) for r in rows) or [0]
    return {
        "prompts": len(rows),
        "conversations": len({r["source"]["example_id"] for r in rows}),
        "tasks": len({(r["domain"], r["task_id"]) for r in rows}),
        "domains": dict(sorted(collections.Counter(r["domain"] for r in rows).items())),
        "kinds": dict(sorted(collections.Counter(r["kind"] for r in rows).items())),
        "tools": dict(collections.Counter(c["name"] for r in rows for c in r["target_calls"]).most_common()),
        "multi_call_turns": sum(len(r["target_calls"]) > 1 for r in rows),
        "approx_prompt_tokens": {
            "median": lengths[len(lengths) // 2],
            "p90": lengths[int(0.9 * (len(lengths) - 1))],
            "max": lengths[-1],
        },
    }


def build(
    train: Sequence[Record], heldout: Sequence[Record], kinds: Sequence[str] = ("call",)
) -> tuple[list[Record], list[Record], Record]:
    """GRPO prompts for training and for the held-out check, plus a report. Raises `GrpoDataError`."""
    bad = [k for k in kinds if k not in KINDS]
    if bad or not kinds:
        raise GrpoDataError(f"kinds must be some of {KINDS}, got {list(kinds)}")
    problems = [p for e in [*train, *heldout] for p in problems_of(e)]
    if problems:
        raise GrpoDataError(*problems[:20], f"{len(problems)} problem(s): nothing written")
    ids = collections.Counter(e["id"] for e in [*train, *heldout])
    if dup := sorted(i for i, n in ids.items() if n > 1):
        raise GrpoDataError(f"conversation(s) given twice: {', '.join(dup[:5])}")
    tasks = {(e["domain"], str(e["task_id"])) for e in train}
    if both := sorted(tasks & {(e["domain"], str(e["task_id"])) for e in heldout}):
        raise GrpoDataError(
            f"{len(both)} task(s) in both train and held out, e.g. {both[0][0]}/{both[0][1]}: "
            "a held-out task must be unseen"
        )
    train_rows = [t for e in train for t in turns_of(e, kinds)]
    heldout_rows = [t for e in heldout for t in turns_of(e, kinds)]
    if not train_rows:
        raise GrpoDataError(f"no {'/'.join(kinds)} turns to train on")
    report = {"kinds": list(kinds), "train": _summary(train_rows), "heldout": _summary(heldout_rows)}
    return train_rows, heldout_rows, report
