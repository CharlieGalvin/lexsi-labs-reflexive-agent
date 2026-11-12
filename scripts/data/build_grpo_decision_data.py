"""Decision GRPO prompts (arm H2): H's heal prompts plus the teacher's decision points, each with
the expected state changes still pending at that turn.

    just grpo-decision-data                          # → data/train/grpo-decision*.jsonl
    just grpo-decision-data --decisions-per-fix 1.5  # more decision prompts per heal fix

Starts from exactly H's data (`reflexive.heal_data.build`: the teacher's fixes at Tool / Arg / Loop
failures, anchors, the same held-out tasks), then adds **decision prompts**: the teacher's turns
where it makes a state change (and the turn just before), from its verified train conversations,
labelled ``DECISION``. Every row gets a ``decision`` column (`reflexive.decision_data`): the
expected changes not yet made in its prompt, and the domain's state-changing tools.

**Never a validation, dev or test task:** the validation tasks are read from arm P's
``P-<domain>-val.jsonl`` (``--val-from``), and the script refuses to run without them.

Writes ``data/train/grpo-decision.jsonl``, ``grpo-decision.heldout.jsonl`` and
``grpo-decision.report.json``, with a check: the teacher's own turn scored by the decision
reward (it should be 1.0 nearly everywhere) and the student's failed turns (lower).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from reflexive import decision_data as dd
from reflexive import grpo_data, heal_data
from reflexive.files import ROOT, read_jsonl, sha256, shown
from reflexive.heal_data import failed_text

FIXES = ROOT / "data" / "heal" / "fixes.jsonl"
GOLDEN = ROOT / "data" / "train" / "sft-healable.jsonl"
VAL_FROM = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "data" / "train"
NAME = "grpo-decision"


def val_tasks(folder: Path) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for d in ("retail", "airline", "telecom"):
        path = folder / f"P-{d}-val.jsonl"
        if not path.exists():
            raise SystemExit(f"⛔ no {shown(path)}: needed to keep the validation tasks out (--val-from)")
        found |= {(r["domain"], str(r["task_id"])) for r in read_jsonl(path, partial_last_ok=True)}
    return found


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The decision reward on the teacher's own turn and on the student's failed turn."""
    out: dict[str, Any] = {}
    for label in sorted({r["failure_type"] for r in rows}):
        mine = [r for r in rows if r["failure_type"] == label]
        teacher = [dd.decision_score(r["target_calls"], r["target_calls"], r["decision"])[0] for r in mine]
        failed = [
            dd.decision_score(grpo_data.calls_of(failed_text(r["failed"])), r["target_calls"], r["decision"])[
                0
            ]
            for r in mine
            if r["failed"].get("calls") or r["failed"].get("text")
        ]
        out[label] = {
            "prompts": len(mine),
            "with_pending_changes": sum(bool(r["decision"]["pending"]) for r in mine),
            "teacher_makes_a_change": sum(
                any(c["name"] in r["decision"]["writes"] for c in r["target_calls"]) for r in mine
            ),
            "teacher_decision_mean": round(sum(teacher) / len(teacher), 3) if teacher else None,
            "teacher_below_1": sum(t < 1 for t in teacher),
            "failed_turn_decision_mean": round(sum(failed) / len(failed), 3) if failed else None,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixes", type=Path, default=FIXES, help=f"default: {shown(FIXES)}")
    p.add_argument("--golden", type=Path, default=GOLDEN, help=f"default: {shown(GOLDEN)}")
    p.add_argument("--val-from", type=Path, default=VAL_FROM, help="folder with P-<domain>-val.jsonl")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--anchors-per-fix", type=float, default=0.5)  # as H
    p.add_argument("--decisions-per-fix", type=float, default=1.0)
    p.add_argument("--heldout-fraction", type=float, default=0.1)  # as H
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    for path, how in ((a.fixes, "just heal-fixes"), (a.golden, "just train-data")):
        if not path.exists():
            print(f"⛔ no {shown(path)}: run `{how}` first")
            return 1
    banned = val_tasks(a.val_from)
    fixes = [e for e in read_jsonl(a.fixes) if (e["domain"], str(e["task_id"])) not in banned]
    golden = [e for e in read_jsonl(a.golden) if (e["domain"], str(e["task_id"])) not in banned]
    dropped = {
        "fixes_on_validation_tasks": len(read_jsonl(a.fixes)) - len(fixes),
        "golden_on_validation_tasks": len(read_jsonl(a.golden)) - len(golden),
    }
    try:
        train, held, report = heal_data.build(
            fixes,
            golden,
            anchors_per_fix=a.anchors_per_fix,
            heldout_fraction=a.heldout_fraction,
            seed=a.seed,
        )
    except heal_data.HealDataError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    held_tasks = {tuple(t.split("/", 1)) for t in report["heldout_tasks"]}
    n_fix = {
        side: sum(r["failure_type"] in heal_data.TYPES for r in rows)
        for side, rows in (("train", train), ("heldout", held))
    }
    golden_tasks = {(e["domain"], str(e["task_id"])) for e in golden}
    for side, rows, exclude in (
        ("train", train, held_tasks | banned),
        ("heldout", held, (golden_tasks - held_tasks) | banned),
    ):
        ids = {r["id"] for r in rows}
        extra = dd.decision_rows(
            golden, exclude_tasks=exclude, n=round(a.decisions_per_fix * n_fix[side]) + len(rows), seed=a.seed
        )
        extra = [r for r in extra if r["id"] not in ids][: round(a.decisions_per_fix * n_fix[side])]
        rows += extra
    for r in [*train, *held]:
        r["decision"] = dd.decision_column(r["domain"], r["task_id"], r["prompt"])
    for t in (*train, *held):
        if (t["domain"], str(t["task_id"])) in banned:
            print(f"⛔ {t['id']} is a validation task: refusing to write")
            return 1

    import random

    random.Random(a.seed).shuffle(train)
    a.out.mkdir(parents=True, exist_ok=True)
    out, held_out = a.out / f"{NAME}.jsonl", a.out / f"{NAME}.heldout.jsonl"
    write_jsonl(out, train)
    write_jsonl(held_out, held)
    counts = {
        side: dict(collections.Counter(r["failure_type"] for r in rows).most_common())
        for side, rows in (("train", train), ("heldout", held))
    }
    full = {
        "format": heal_data.FORMAT,
        "inputs": {
            "fixes": {"file": shown(a.fixes), "sha256": sha256(a.fixes)},
            "golden": {"file": shown(a.golden), "sha256": sha256(a.golden)},
            "validation_tasks_excluded": len(banned),
            **dropped,
        },
        "failure_type": counts,
        "check": {"train": check(train), "heldout": check(held)},
        "heldout_tasks": report["heldout_tasks"],
        "anchors_per_fix": a.anchors_per_fix,
        "decisions_per_fix": a.decisions_per_fix,
        "seed": a.seed,
        "files": {out.name: sha256(out), held_out.name: sha256(held_out)},
    }
    (a.out / f"{NAME}.report.json").write_text(json.dumps(full, indent=1) + "\n")
    for side in ("train", "heldout"):
        print(f"{side}: {sum(counts[side].values())} prompts: {counts[side]}")
    for label, c in full["check"]["train"].items():
        print(
            f"  {label}: teacher decision {c['teacher_decision_mean']} ({c['teacher_below_1']} below 1), "
            f"failed turn {c['failed_turn_decision_mean']}, pending at {c['with_pending_changes']}, "
            f"teacher changes state at {c['teacher_makes_a_change']}"
        )
    print(f"wrote {shown(out)}, {held_out.name} and {NAME}.report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
