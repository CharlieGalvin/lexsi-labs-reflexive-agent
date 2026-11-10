"""Run an arm on the 30 validation tasks, 2 samples each, the way SFT v2, v3 and v4 were scored.

    just arm-val G configs/sft/arms.grpo.yaml                 # retail, airline, telecom in turn
    just arm-val G configs/sft/arms.grpo.yaml --dry-run       # the task lists only
    just arm-val G configs/sft/arms.grpo.yaml --from <folder with P-*-val.jsonl>
    just arm-samples S2 G --split val                         # then: totals per domain and sample

Validation isn't a split in ``configs/splits.json``: it's 30 train tasks no arm trained on (the 15
held-out ``solved_by_aya`` tasks and the 15 without a teacher fix; ``docs/experiments/SFT.md``).
Each domain's task list is read from ``P-<domain>-val.jsonl`` (arm P's validation run) in
``--from``, and ``scripts/arms/rollout.py`` writes ``results/episodes/arms/<arm>-<domain>-val.jsonl``.
A re-run resumes: finished conversations are kept and crashed ones run again.

Refuses a task outside the train split, and (if ``data/train/grpo-healable.jsonl`` is there) a task
GRPO trained on.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reflexive import splits
from reflexive.files import ROOT, read_jsonl, shown

DOMAINS = ("retail", "airline", "telecom")
ARMS_DIR = ROOT / "results" / "episodes" / "arms"
GRPO_DATA = ROOT / "data" / "train" / "grpo-healable.jsonl"
SAMPLES = 2


def val_tasks(source: Path, domain: str) -> list[str]:
    """The validation task IDs of one domain, from arm P's validation file, sorted."""
    path = source / f"P-{domain}-val.jsonl"
    if not path.exists():
        raise SystemExit(
            f"⛔ no {shown(path)}: copy P-*-val.jsonl from the pod that ran arm P on validation, "
            "or pass --from with its folder"
        )
    ids = sorted({str(r["task_id"]) for r in read_jsonl(path, partial_last_ok=True)})
    if not ids:
        raise SystemExit(f"⛔ {shown(path)} has no tasks")
    for t in ids:
        splits.assert_not_test(domain, t)
        if splits.split_of(domain, t) != "train":
            raise SystemExit(f"⛔ {domain}/{t} in {shown(path)} isn't a train task: not a validation task")
    return ids


def trained_tasks(path: Path = GRPO_DATA) -> set[tuple[str, str]]:
    """The (domain, task) pairs GRPO trained on, if its data file is here."""
    if not path.exists():
        return set()
    return {(r["domain"], str(r["task_id"])) for r in read_jsonl(path)}


def command(arm: str, domain: str, ids: list[str], config: Path, samples: int, extra: list[str]) -> list[str]:
    out = ARMS_DIR / f"{arm}-{domain}-val.jsonl"
    return [
        sys.executable,
        str(ROOT / "scripts" / "arms" / "rollout.py"),
        arm,
        domain,
        "--task-ids",
        *ids,
        "--samples",
        str(samples),
        "--config",
        str(config),
        "--out",
        str(out),
        *extra,
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", help="the arm's key in --config, e.g. G")
    p.add_argument("config", type=Path, help="the arms config, e.g. configs/sft/arms.grpo.yaml")
    p.add_argument(
        "--from", dest="source", type=Path, default=ARMS_DIR, help="where P-<domain>-val.jsonl are"
    )
    p.add_argument("--samples", type=int, default=SAMPLES, help=f"conversations per task (default {SAMPLES})")
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    p.add_argument("--dry-run", action="store_true", help="print the task lists and commands only")
    a, extra = p.parse_known_args(argv)  # anything else goes to rollout.py (e.g. --workers 8)

    tasks = {d: val_tasks(a.source, d) for d in a.domains}
    if both := sorted((d, t) for d, ids in tasks.items() for t in ids if (d, t) in trained_tasks()):
        raise SystemExit(f"⛔ {len(both)} validation task(s) are in GRPO's training data, e.g. {both[0]}")
    print(
        f"Arm {a.arm} on validation: "
        + ", ".join(f"{d} {len(ids)}" for d, ids in tasks.items())
        + f" tasks ({sum(map(len, tasks.values()))}), {a.samples} samples each"
    )
    for d, ids in tasks.items():
        cmd = command(a.arm, d, ids, a.config, a.samples, extra)
        if a.dry_run:
            print("   " + " ".join(cmd[1:]))
            continue
        if (code := subprocess.run(cmd, check=False).returncode) != 0:
            print(f"⛔ {d} stopped (exit {code}): re-run the same command to resume")
            return code
    if not a.dry_run:
        print(f"done: just arm-samples S2 {a.arm} --split val")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
