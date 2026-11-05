"""Summarise an arm's rollout file as a ``.metrics.json`` (SEL-55). No API calls.

    just arm-metrics results/episodes/arms/A-retail-train.jsonl
    just arm-metrics results/episodes/arms/A-telecom-train.jsonl --only dev   # its dev tasks only
    python scripts/arms/metrics.py <file> --note "…"   # a note with spaces or brackets: `just` drops quotes

Writes ``results/arm-<arm>-<domain>-<split>.metrics.json`` (committed: `results/` keeps
``*.metrics.json``), in the format of Chiara's first ones (4 Nov). The numbers are defined in
``reflexive.metrics``. ``--only dev`` or ``--only train`` keeps the records whose task is in that
frozen split (``configs/splits.json``): τ²-bench's own train split, which an older run may have
used, holds both.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import ROOT, BadJsonLine, read_jsonl, shown
from reflexive.metrics import arm_metrics

STUDENT = "c4ai-aya-expanse-32b (Cohere API, prompted, cohere Action: format)"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def header(recs: list[dict[str, Any]], path: Path, only: str | None, owner: str) -> dict[str, Any]:
    """Who ran what, where: the fields before the numbers."""
    first = recs[0]
    arm, domain = first.get("arm", "?"), first["domain"]
    file_split = path.stem.split("-", 2)[-1]  # A-retail-train → train
    split = only or file_split
    judged = any(r["outcome"].get("judged") for r in recs)
    return {
        "run": f"arm-{arm}-{domain}-{split}",
        "date": dt.date.today().isoformat(),
        "owner": owner,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "config": "configs/arms.yaml",
        "arm": arm,
        "prompt_variant": first.get("prompt_variant"),
        "prompt_version": first.get("prompt_version"),
        "student": STUDENT if first.get("agent_llm") == "c4ai-aya-expanse-32b" else first.get("agent_llm"),
        "customer": first.get("user_llm"),
        "domain": domain,
        "split": f"{split} (frozen splits, configs/splits.json)",
        "samples_per_task": max(r.get("sample", 0) for r in recs) + 1,
        "judges": judged,
        "note_judges": (
            "telecom has no LLM judges: scores are complete"
            if domain == "telecom"
            else "τ²-bench's LLM judges ran (OPENAI_API_KEY set)"
            if judged
            else "deterministic checks only: the LLM-judged checks were skipped"
        ),
        "records": f"{shown(path)} (not committed)" + (f", the {only} tasks only" if only else ""),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "records", type=Path, help="a rollout file, results/episodes/arms/<arm>-<domain>-<split>.jsonl"
    )
    p.add_argument("--only", choices=["train", "dev"], help="keep only this frozen split's tasks")
    p.add_argument("--owner", default=_git("config", "user.name"), help="who ran it (default: git user.name)")
    p.add_argument("--note", help="anything a reader should know, saved as `history`")
    p.add_argument("--out", type=Path, help="default: results/arm-<arm>-<domain>-<split>.metrics.json")
    a = p.parse_args()

    try:
        recs = read_jsonl(a.records)
    except (BadJsonLine, FileNotFoundError) as e:
        print(f"⛔ {e}")
        return 1
    if not recs:
        print(f"⛔ {shown(a.records)} has no records")
        return 1
    if a.only:
        recs = [r for r in recs if splits.split_of(r["domain"], str(r["task_id"])) == a.only]
        if not recs:
            print(f"⛔ no {a.only} tasks in {shown(a.records)}")
            return 1
    test = [r["task_id"] for r in recs if splits.split_of(r["domain"], str(r["task_id"])) == "test"]
    if test:
        print(f"⛔ {len(test)} test-split tasks in the file (e.g. {test[0]}): summarise test runs separately")
        return 1

    out = {**header(recs, a.records, a.only, a.owner), **arm_metrics(recs)}
    if a.note:
        out["history"] = a.note
    path = a.out or ROOT / "results" / f"{out['run']}.metrics.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    lo, hi = out["task_success_wilson95"]
    print(
        f"{out['run']}: {out['passed']}/{out['conversations']} passed ({lo:.0%}–{hi:.0%}), "
        f"{out['crashed']} crashed, ${out['cost_usd']['total']:.2f} → {shown(path)}"
    )
    if out["crashed"]:
        print("⚠ crashed conversations aren't scored: re-run the arm to retry them, then summarise again")
    return 0


if __name__ == "__main__":
    sys.exit(main())
