"""Training examples from golden trajectories v1.1 (SEL-66). No API calls, no GPU.

    just train-data                                   # local builds: data/golden/v1 + data/golden/hard
    just train-data --dataset data/golden/hub-v1.1    # the HF snapshot at tag v1.1 (on the pod)
    just train-data --select all                      # all 127 verified fixes, not only the 112 healable

Writes ``data/train/sft-<select>.jsonl`` (one conversation per line: ``messages`` as Aya reads
them, ``train_on`` = the agent's turns, the only learning targets) and ``sft-<select>.report.json``
(counts, lengths, checksums of the inputs and output). See ``reflexive.train_data`` for the
checks; any failure writes nothing. Refuses to replace an existing build without ``--force``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from reflexive import train_data
from reflexive.files import ROOT, BadJsonLine, read_jsonl, sha256, shown

GOLDEN = ROOT / "data" / "golden" / "v1" / "golden_train.jsonl"
HARD = ROOT / "data" / "golden" / "hard" / "hard_subset.jsonl"
OUT = ROOT / "data" / "train"


def inputs(a: argparse.Namespace) -> tuple[Path, Path]:
    if a.dataset:
        return a.dataset / GOLDEN.name, a.dataset / HARD.name
    return a.golden, a.hard


def check_built_together(golden: Path, hard: Path) -> None:
    """If the hard subset's build report is beside it, it must have been built from this golden file."""
    report = hard.parent / "hard_report.json"
    if not report.exists():
        return  # e.g. the HF snapshot: the tag already ties the two together
    try:
        built_from = json.loads(report.read_text())["inputs"]["golden"]["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise train_data.TrainDataError(f"can't read {shown(report)} ({type(e).__name__})") from e
    if built_from != sha256(golden):
        raise train_data.TrainDataError(
            f"{shown(hard)} was built from another {golden.name}: rebuild it (`just hard-subset --force`)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--golden", type=Path, default=GOLDEN, help="golden_train.jsonl")
    p.add_argument("--hard", type=Path, default=HARD, help="hard_subset.jsonl")
    p.add_argument("--dataset", type=Path, help="a folder holding both (e.g. the HF snapshot at v1.1)")
    p.add_argument("--select", choices=train_data.SELECTIONS, default="healable")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="replace an existing build")
    a = p.parse_args()

    golden, hard = inputs(a)
    out = a.out / f"sft-{a.select}.jsonl"
    if out.exists() and not a.force:
        print(f"⛔ {shown(out)} exists: pass --force to rebuild it")
        return 1
    try:
        for f in (golden, hard):
            if not f.exists():
                raise train_data.TrainDataError(f"{shown(f)} doesn't exist")
        check_built_together(golden, hard)
        examples, report = train_data.build(read_jsonl(golden), read_jsonl(hard), a.select)
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return 1
    except train_data.TrainDataError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples))
    full: dict[str, Any] = {
        "format": train_data.FORMAT,
        "built": dt.date.today().isoformat(),
        "inputs": {
            "golden": {"file": shown(golden), "sha256": sha256(golden)},
            "hard": {"file": shown(hard), "sha256": sha256(hard)},
        },
        **report,
        "files": {out.name: sha256(out)},
    }
    out.with_suffix(".report.json").write_text(json.dumps(full, indent=1) + "\n")
    print(
        f"{report['examples']} examples ({a.select}): "
        + ", ".join(f"{d} {n}" for d, n in report["domains"].items())
        + f"; {report['agent_turns']} agent turns to learn from"
    )
    print(
        f"labels: {', '.join(f'{k} {v}' for k, v in report['labels'].items())}; "
        f"longest ≈ {report['approx_tokens']['max']:,} tokens, total ≈ {report['approx_tokens']['total']:,}"
    )
    print(f"wrote {shown(out)} and {out.with_suffix('.report.json').name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
