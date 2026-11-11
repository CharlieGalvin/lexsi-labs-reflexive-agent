"""Heal GRPO prompts from the teacher's fixes, plus anchor turns (SEL-75).

    just grpo-heal-data                         # fixes + anchors → data/train/grpo-heal*.jsonl
    just grpo-heal-data --anchors-per-fix 0.5   # fewer ordinary turns

Each accepted fix (`just heal-fixes`) becomes one prompt: the student's conversation up to its
failing turn, with the teacher's fix as the reference turn (a tool call or a reply), labelled
Tool / Arg / Loop. **Anchors** are ordinary turns from the teacher's own conversations
(``just train-data``: replies as well as tool calls), ``--anchors-per-fix`` per fix, labelled
``NONE``. A seeded 10% of the fixes' tasks is held out (fixes and anchors) for ``just grpo-heldout``.

Writes ``data/train/grpo-heal.jsonl``, ``grpo-heal.heldout.jsonl`` and ``grpo-heal.report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from reflexive import heal_data
from reflexive.files import ROOT, read_jsonl, sha256, shown

FIXES = ROOT / "data" / "heal" / "fixes.jsonl"
GOLDEN = ROOT / "data" / "train" / "sft-healable.jsonl"
OUT = ROOT / "data" / "train"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixes", type=Path, default=FIXES, help=f"default: {shown(FIXES)}")
    p.add_argument("--golden", type=Path, default=GOLDEN, help=f"anchors from (default: {shown(GOLDEN)})")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--anchors-per-fix", type=float, default=1.0)
    p.add_argument("--heldout-fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    for path, how in ((a.fixes, "just heal-fixes"), (a.golden, "just train-data")):
        if not path.exists():
            print(f"⛔ no {shown(path)}: run `{how}` first")
            return 1
    try:
        train, held, report = heal_data.build(
            read_jsonl(a.fixes),
            read_jsonl(a.golden),
            anchors_per_fix=a.anchors_per_fix,
            heldout_fraction=a.heldout_fraction,
            seed=a.seed,
        )
    except heal_data.HealDataError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    out, held_out = a.out / "grpo-heal.jsonl", a.out / "grpo-heal.heldout.jsonl"
    write_jsonl(out, train)
    write_jsonl(held_out, held)
    full = {
        "format": heal_data.FORMAT,
        "inputs": {
            "fixes": {"file": shown(a.fixes), "sha256": sha256(a.fixes)},
            "golden": {"file": shown(a.golden), "sha256": sha256(a.golden)},
        },
        **report,
        "files": {out.name: sha256(out), held_out.name: sha256(held_out)},
    }
    (a.out / "grpo-heal.report.json").write_text(json.dumps(full, indent=1) + "\n")
    for label, r in (("train", report["train"]), ("held out", report["heldout"])):
        print(f"{label}: {r['prompts']} prompts from {r['tasks']} tasks; {r['failure_type']}; {r['kind']}")
    print(f"wrote {shown(out)}, {held_out.name} and grpo-heal.report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
