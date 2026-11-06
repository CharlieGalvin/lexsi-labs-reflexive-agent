"""The hard subset (SEL-7): every train task, grouped by how Arm A did and whether a golden fix exists.

Built from Arm A's train runs (SEL-55, ``results/episodes/arms/A-<domain>-train.jsonl``, one
sample per task) and golden trajectories v1 (``data/golden/v1/golden_train.jsonl``):

* ``healable``: Arm A failed **and** the teacher's verified fix exists. What healing trains on.
* ``teacher_unsolved``: Arm A failed and the teacher never solved it: beyond the teacher, for now.
* ``solved_by_aya``: Arm A passed. Checks that healing doesn't break what already works.

Each task also gets **failure labels**: every failure type the classifier (SEL-12,
``classify_record``) finds in Arm A's conversation, ``TOOL``, ``ARG`` and ``LOOP``, or ``NONE``
when it finds none (the conversation failed for a reasoning or policy reason the classifier
doesn't judge). In telecom, ``customer_tool`` marks a call to one of the customer's own phone
tools. A passing conversation can carry labels too: the classifier finds structural slips, and a
conversation can recover from them.

Only **train** tasks are kept. Dev tasks (telecom's Arm A file holds its 15 dev tasks) are left
out and counted, and a test task anywhere stops the build. Stdlib only: no τ²-bench needed.
"""

from __future__ import annotations

import collections
from typing import Any

from reflexive import splits
from reflexive.hallucination import FailureType, agent_view_of, classify_record
from reflexive.runner import is_broken

FORMAT = "reagent.hard/v1"
GROUPS = ("healable", "teacher_unsolved", "solved_by_aya")
LABELS = ("TOOL", "ARG", "LOOP", "NONE")
MIN_EXAMPLES = 5  # fewer healable tasks than this with a label: too thin to train that type on

Record = dict[str, Any]


class HardSubsetError(Exception):
    """A check failed: nothing is written. ``lines`` are printed with ⛔."""

    def __init__(self, *lines: str) -> None:
        super().__init__(lines[-1])
        self.lines = lines


def labels_of(rec: Record) -> tuple[list[str], dict[str, Any]]:
    """The failure types in Arm A's conversation (TOOL, ARG, LOOP order, or ``["NONE"]``), and the detail."""
    d = classify_record(rec)
    types = [t.value for t in (FailureType.TOOL, FailureType.ARG, FailureType.LOOP) if t in d.types]
    detail = {
        "primary": d.failure.value,
        "first": d.first.to_dict() if d.first else None,
        "kinds": dict(d.kinds),
        "customer_tool": any(f.kind == "customer_tool" for f in d.findings),
    }
    return types or ["NONE"], detail


def train_records(arm: list[Record]) -> tuple[dict[tuple[str, str], Record], collections.Counter]:
    """Arm A's train records by (domain, task), and how many dev records were left out per domain.

    Stops on a test task, a task in no split, a broken run or more than one sample of a task.
    """
    kept: dict[tuple[str, str], Record] = {}
    dev_tasks: set[tuple[str, str]] = set()
    problems = []
    for r in arm:
        key = (r["domain"], str(r["task_id"]))
        try:
            split = splits.split_of(*key)
        except KeyError:
            problems.append(f"{key[0]} task {key[1]!r} is in no split")
            continue
        if split == "test":
            problems.append(f"{key[0]} task {key[1]!r} is a TEST task: it must never be in the hard subset")
        elif split == "dev":
            dev_tasks.add(key)
        elif key in kept:
            problems.append(
                f"{key[0]} task {key[1]}: more than one sample; the hard subset uses one sample per task"
            )
        elif is_broken(r):
            problems.append(f"{key[0]} task {key[1]}: the run broke; re-run it (`just rollout A {key[0]}`)")
        elif not isinstance(r.get("outcome"), dict) or "passed" not in r["outcome"]:
            problems.append(f"{key[0]} task {key[1]}: no saved outcome")
        else:
            kept[key] = r
    if problems:
        raise HardSubsetError(*problems, f"{len(problems)} problems in Arm A's records; nothing written")
    return kept, collections.Counter(d for d, _ in dev_tasks)


def golden_ids(golden: list[Record]) -> dict[tuple[str, str], str]:
    """The golden fix's record ID per (domain, task). Stops on a non-train record or two fixes for a task."""
    ids: dict[tuple[str, str], str] = {}
    for g in golden:
        key = (g["domain"], str(g["task_id"]))
        try:
            split = splits.split_of(*key)
        except KeyError:
            split = "no split"
        if g.get("split") != "train" or split != "train":
            raise HardSubsetError(f"golden record {g['id']} isn't a train task")
        if key in ids:
            raise HardSubsetError(f"two golden fixes for {key[0]} task {key[1]}: {ids[key]}, {g['id']}")
        ids[key] = g["id"]
    return ids


def group_of(passed: bool, has_fix: bool) -> str:
    if passed:
        return "solved_by_aya"
    return "healable" if has_fix else "teacher_unsolved"


def to_record(rec: Record, golden_id: str | None, source: dict[str, str]) -> Record:
    """One hard-subset record: the task, its group, Arm A's result and labels, and its conversation."""
    labels, detail = labels_of(rec)
    out = rec["outcome"]
    passed = bool(out["passed"])
    return {
        "id": f"armA/{rec['domain']}/{rec['task_id']}",
        "format": FORMAT,
        "domain": rec["domain"],
        "task_id": str(rec["task_id"]),
        "split": "train",
        "group": group_of(passed, golden_id is not None),
        "golden_id": golden_id,
        "labels": labels,
        **detail,
        "arm_a": {
            "samples": 1,
            "passed": int(passed),
            "reward": out.get("reward"),
            "checks": out.get("checks"),
            "termination": rec.get("termination"),
            "student": rec.get("student"),
            "source": source,
        },
        "messages": agent_view_of(rec),  # OpenAI format, as Arm A saw it: context, never a learning target
    }


def build(
    arm: list[Record], golden: list[Record], sources: dict[str, dict[str, str]]
) -> tuple[list[Record], dict[str, Any]]:
    """The hard-subset records, in frozen train order, and the counts for the build report.

    ``sources`` maps each domain to the Arm A file it came from (``{"file", "sha256_16"}``).
    Every train task of every domain in ``sources`` must have exactly one Arm A record.
    """
    kept, dev = train_records(arm)
    fixes = golden_ids(golden)
    domains = [d for d in splits.DOMAINS if d in sources]
    missing = [
        f"{d} task {t}: no Arm A record"
        for d in domains
        for t in splits.task_ids(d, "train")
        if (d, t) not in kept
    ]
    extra = sorted({d for d, _ in kept} - set(domains))
    if missing or extra:
        raise HardSubsetError(
            *missing[:20],
            *(f"{d}: records but no source file" for d in extra),
            f"{len(missing)} train tasks without an Arm A record, {len(extra)} domains without a source; "
            "nothing written",
        )
    records = [
        to_record(kept[(d, t)], fixes.get((d, t)), sources[d])
        for d in domains
        for t in splits.task_ids(d, "train")
    ]
    return records, counts(records, dev)


def counts(records: list[Record], dev_left_out: collections.Counter) -> dict[str, Any]:
    """Tasks per group, and label counts per group, for each domain and in total."""

    def tally(recs: list[Record]) -> dict[str, Any]:
        return {
            "tasks": len(recs),
            "groups": {g: sum(r["group"] == g for r in recs) for g in GROUPS},
            "labels": {
                g: {lab: sum(lab in r["labels"] for r in recs if r["group"] == g) for lab in LABELS}
                for g in GROUPS
            },
            "primary": {
                g: dict(collections.Counter(r["primary"] for r in recs if r["group"] == g)) for g in GROUPS
            },
            "customer_tool": {g: sum(r["customer_tool"] for r in recs if r["group"] == g) for g in GROUPS},
        }

    domains = list(dict.fromkeys(r["domain"] for r in records))
    total = tally(records)
    healable = total["labels"]["healable"]
    return {
        "domains": {d: tally([r for r in records if r["domain"] == d]) for d in domains},
        "total": total,
        "dev_left_out": {d: dev_left_out.get(d, 0) for d in domains},
        "thin": [lab for lab in ("TOOL", "ARG", "LOOP") if healable[lab] < MIN_EXAMPLES],
        "min_examples": MIN_EXAMPLES,
    }
