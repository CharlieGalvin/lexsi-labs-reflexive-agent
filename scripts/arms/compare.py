"""Compare two τ²-bench rollout files, with 95% bootstrap confidence intervals (SEL-27). No API calls.

    just arm-compare results/episodes/arms/A-telecom-train.jsonl results/episodes/arms/C-telecom-train.jsonl
    python scripts/arms/compare.py <a.jsonl> <b.jsonl> --names A C --reps 20000

For success rate, hallucination rate (Tool or Arg), loop rate, steps per task and redundant-call
rate: each run's value with its 95% range, and the difference b − a, paired on the tasks both
ran. Definitions and method: `reflexive.compare`.

Prints the table and writes it to ``results/<a>-vs-<b>.compare.md``, plus the numbers to
``results/<a>-vs-<b>.compare.metrics.json`` (committed, as the other ``*.metrics.json`` are).
Names default to the files' names without ``.jsonl``. A half-written last line (a stopped
rollout) is skipped. Refuses two runs with no task in common, and warns when they share only
some of their tasks (the difference then covers the shared ones only).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from reflexive.compare import REPS, compare, to_markdown
from reflexive.files import ROOT, read_jsonl

RESULTS = ROOT / "results"


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path, help="the first run's rollout file (the baseline)")
    ap.add_argument("b", type=Path, help="the second run's rollout file")
    ap.add_argument("--names", nargs=2, metavar=("A", "B"), help="names for the two runs")
    ap.add_argument("--reps", type=int, default=REPS, help=f"bootstrap resamples (default {REPS:,})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="print only")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not p.exists():
            ap.error(f"no such file: {p}")
    name_a, name_b = args.names or (args.a.name.removesuffix(".jsonl"), args.b.name.removesuffix(".jsonl"))
    a = read_jsonl(args.a, partial_last_ok=True)
    b = read_jsonl(args.b, partial_last_ok=True)
    tasks_a = {(str(r["domain"]), str(r["task_id"])) for r in a}
    tasks_b = {(str(r["domain"]), str(r["task_id"])) for r in b}
    shared = tasks_a & tasks_b
    if not shared:
        ap.error(f"{name_a} and {name_b} have no task in common: there's nothing to pair")
    if tasks_a != tasks_b:
        print(
            f"⚠ {name_a} ran {len(tasks_a)} tasks, {name_b} {len(tasks_b)}; the difference covers the "
            f"{len(shared)} both ran",
            file=sys.stderr,
        )

    try:
        c = compare(a, b, reps=args.reps, seed=args.seed)
    except ValueError as e:
        ap.error(str(e))
    table = to_markdown(c, name_a, name_b)
    print(table)
    if args.no_write:
        return 0

    stem = f"{name_a}-vs-{name_b}"
    head = (
        f"# {name_a} vs {name_b}\n\nτ²-bench rollouts: [`{args.a}`] vs [`{args.b}`]. Made by "
        f"`just arm-compare` @ {_commit()} on {dt.date.today()} (`reflexive.compare`).\n\n"
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{stem}.compare.md").write_text(head + table + "\n")
    out = {
        "a": {"name": name_a, "file": str(args.a)},
        "b": {"name": name_b, "file": str(args.b)},
        "commit": _commit(),
        "date": str(dt.date.today()),
        **c.to_dict(),
    }
    (RESULTS / f"{stem}.compare.metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"\n→ results/{stem}.compare.md, results/{stem}.compare.metrics.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
