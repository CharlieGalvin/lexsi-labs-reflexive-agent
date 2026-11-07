"""Training examples from golden trajectories v1.1 (SEL-66): what the student learns from.

Golden v1 is already stored in the format Aya reads at run time (the data contract, SEL-35):
τ²-bench's system prompt with the tools as Python stubs, tool calls as ``Action: ```json [...]```
``, results as ``<results>`` user turns, and ``train_on`` = every assistant turn. So nothing is
re-rendered here; this module picks the fixes, checks each one strictly, and writes them out:

* ``select="healable"`` (default): the fixes for tasks Arm A failed (the hard subset's
  ``healable`` group, SEL-7). What healing trains on.
* ``select="all"``: every verified fix, adding the 15 for tasks Aya already passes.

Each example: ``messages`` (system, then alternating assistant / user turns, ending on the
customer's sign-off, ``###STOP###`` or ``###TRANSFER###``, as τ²-bench ends them), ``train_on`` (the
indices of the assistant turns: the only learning targets; the customer's turns are GPT-4.1
outputs, context only, never a target), plus the task, its hard-subset group and failure labels
(for weighting by type). Only **train** tasks: a dev or test task stops the build.

Stdlib only: no τ²-bench, no GPU.
"""

from __future__ import annotations

import collections
from typing import Any

from reflexive import golden, splits
from reflexive.teacher import parse_prompted

FORMAT = "reagent.sft/v1"
SELECTIONS = ("healable", "all")

Record = dict[str, Any]


class TrainDataError(Exception):
    """A check failed: nothing is written. ``lines`` are printed with ⛔."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[-1])
        self.lines = lines


def problems_of(g: Record) -> list[str]:
    """Why a golden record can't be trained on as it stands (empty if it can)."""
    rid = g.get("id", "?")
    out = []
    if g.get("format") != golden.FORMAT:
        out.append(f"{rid}: format {g.get('format')!r}, expected {golden.FORMAT}")
    try:
        split = splits.split_of(g["domain"], str(g["task_id"]))
    except KeyError:
        return [*out, f"{rid}: its task is in no split"]
    if split != "train" or g.get("split") != "train":
        out.append(f"{rid}: a {split} task; only train tasks are trained on")
    msgs = g.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        return [*out, f"{rid}: doesn't start with the system prompt"]
    roles = [m.get("role") for m in msgs[1:]]
    if any(r not in ("assistant", "user") for r in roles):
        out.append(f"{rid}: a turn that isn't assistant or user")
    if any(a == b for a, b in zip(roles, roles[1:], strict=False)):
        out.append(f"{rid}: two turns in a row from the same role")
    agent = [i for i, m in enumerate(msgs) if m.get("role") == "assistant"]
    if g.get("train_on") != agent:
        out.append(f"{rid}: train_on isn't exactly the agent's turns")
    for i in agent:
        text = msgs[i].get("content") or ""
        if not text.strip():
            out.append(f"{rid}: agent turn {i} is empty")
        elif text.lstrip().startswith("Action:") and not parse_prompted(text):
            out.append(f"{rid}: agent turn {i} is an Action: that doesn't parse")
    return out


def build(
    golden_train: list[Record], hard: list[Record], select: str = "healable"
) -> tuple[list[Record], dict[str, Any]]:
    """The training examples, in the hard subset's (frozen train) order, and their counts."""
    if select not in SELECTIONS:
        raise ValueError(f"select must be one of {', '.join(SELECTIONS)}")
    by_id: dict[str, Record] = {}
    for g in golden_train:
        if g.get("id") in by_id:
            raise TrainDataError(f"golden record {g['id']} appears twice")
        by_id[g.get("id", "?")] = g
    problems: list[str] = []
    picked: list[tuple[Record, Record]] = []
    for h in hard:
        key = (str(h.get("domain")), str(h.get("task_id")))
        try:
            belongs = splits.split_of(*key)
        except KeyError:
            belongs = "no split"
        if h.get("split") != "train" or belongs != "train":
            problems.append(f"hard subset {h.get('id')}: a {belongs} task, not train")
            continue
        gid = h.get("golden_id")
        if gid is None or (select == "healable" and h.get("group") != "healable"):
            continue
        g = by_id.get(gid)
        if g is None:
            problems.append(f"hard subset {h.get('id')}: its fix {gid} isn't in golden_train.jsonl")
            continue
        if (g["domain"], str(g["task_id"])) != key:
            problems.append(f"{gid}: is for {g['domain']} {g['task_id']}, the hard subset says {key}")
            continue
        problems += problems_of(g)
        picked.append((h, g))
    if select == "all":  # every fix must be accounted for by the hard subset
        seen = {g["id"] for _, g in picked}
        problems += [f"{i}: a golden fix the hard subset doesn't list" for i in by_id if i not in seen]
    if problems:
        raise TrainDataError(*problems[:20], f"{len(problems)} problems; nothing written")
    if not picked:
        raise TrainDataError(f"no fixes selected ({select}): is the hard subset empty?")
    examples = [to_example(h, g) for h, g in picked]
    return examples, counts(examples, select)


def to_example(h: Record, g: Record) -> Record:
    return {
        "id": g["id"],
        "format": FORMAT,
        "domain": g["domain"],
        "task_id": str(g["task_id"]),
        "group": h["group"],
        "labels": h.get("labels", []),
        "messages": [{"role": m["role"], "content": m["content"]} for m in g["messages"]],
        "train_on": list(g["train_on"]),
        "source": {"golden_id": g["id"], "hard_id": h["id"], "teacher": g.get("teacher")},
    }


def counts(examples: list[Record], select: str) -> dict[str, Any]:
    """Examples per domain, group and label, and their lengths (to size the context on the GPU)."""
    chars = [sum(len(m["content"]) for m in e["messages"]) for e in examples]
    target = [sum(len(e["messages"][i]["content"]) for i in e["train_on"]) for e in examples]
    labels: collections.Counter[str] = collections.Counter(lab for e in examples for lab in e["labels"])
    return {
        "select": select,
        "examples": len(examples),
        "domains": dict(collections.Counter(e["domain"] for e in examples)),
        "groups": dict(collections.Counter(e["group"] for e in examples)),
        "labels": {lab: labels.get(lab, 0) for lab in ("TOOL", "ARG", "LOOP", "NONE")},
        "turns": sum(len(e["messages"]) for e in examples),
        "agent_turns": sum(len(e["train_on"]) for e in examples),
        # Characters, not tokens (no tokenizer here); about 3.5 characters a token for this text
        "chars": {"max": max(chars), "median": sorted(chars)[len(chars) // 2], "total": sum(chars)},
        "target_chars": sum(target),
        "approx_tokens": {"max": round(max(chars) / 3.5), "total": round(sum(chars) / 3.5)},
    }
