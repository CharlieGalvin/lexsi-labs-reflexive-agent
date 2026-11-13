"""Single-turn GRPO prompts from the training conversations (SEL-21). No API calls, no GPU.

    just grpo-data                                        # healable fixes; solved_by_aya held out
    just grpo-data --train data/train/sft-v2.jsonl        # SFT v2's data instead
    just grpo-data --kinds call text                      # text turns too, not only tool calls

Reads ``reagent.sft/v1`` conversations (``just train-data`` writes them) and writes, in ``--out``:

* ``grpo-<name>.jsonl``: one prompt per agent turn (the conversation up to it, and the teacher's
  turn as the reference), from the ``--train`` conversations;
* ``grpo-<name>.heldout.jsonl``: the same from the held-out conversations: by default the
  ``solved_by_aya`` group of ``--all`` (``sft-all.jsonl``), never trained on;
* ``grpo-<name>.report.json``: counts, prompt lengths, tools, and checksums of the inputs.

``<name>`` is the ``--train`` file's name without ``sft-`` (``healable``, ``v2``, …). See
``reflexive.grpo_data`` for the checks; any failure writes nothing. Refuses to replace an
existing build without ``--force``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

from reflexive import grpo_data
from reflexive.files import ROOT, BadJsonLine, read_jsonl, sha256, shown

TRAIN_DIR = ROOT / "data" / "train"
HELDOUT_GROUP = "solved_by_aya"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--train", type=Path, default=TRAIN_DIR / "sft-healable.jsonl", help="conversations to train on"
    )
    p.add_argument(
        "--all", type=Path, default=TRAIN_DIR / "sft-all.jsonl", help="where the held-out ones come from"
    )
    p.add_argument("--heldout-group", default=HELDOUT_GROUP, help="the group in --all to hold out")
    p.add_argument("--kinds", nargs="+", choices=grpo_data.KINDS, default=["call"])
    p.add_argument("--out", type=Path, default=TRAIN_DIR)
    p.add_argument("--force", action="store_true", help="replace an existing build")
    a = p.parse_args()

    name = a.train.stem.removeprefix("sft-")
    out = a.out / f"grpo-{name}.jsonl"
    heldout_out = a.out / f"grpo-{name}.heldout.jsonl"
    report_out = a.out / f"grpo-{name}.report.json"
    if out.exists() and not a.force:
        print(f"⛔ {shown(out)} exists: pass --force to rebuild it")
        return 1
    try:
        for f in (a.train, a.all):
            if not f.exists():
                raise grpo_data.GrpoDataError(f"{shown(f)} doesn't exist: run `just train-data` first")
        train = read_jsonl(a.train)
        heldout = [e for e in read_jsonl(a.all) if e.get("group") == a.heldout_group]
        if not heldout:
            raise grpo_data.GrpoDataError(f"no {a.heldout_group!r} conversations in {shown(a.all)}")
        train_rows, heldout_rows, report = grpo_data.build(train, heldout, a.kinds)
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return 1
    except grpo_data.GrpoDataError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, train_rows)
    write_jsonl(heldout_out, heldout_rows)
    full = {
        "format": grpo_data.FORMAT,
        "built": dt.date.today().isoformat(),
        "inputs": {
            "train": {"file": shown(a.train), "sha256": sha256(a.train)},
            "heldout": {"file": shown(a.all), "group": a.heldout_group, "sha256": sha256(a.all)},
        },
        **report,
        "files": {out.name: sha256(out), heldout_out.name: sha256(heldout_out)},
    }
    report_out.write_text(json.dumps(full, indent=1) + "\n")
    for label, r in (("train", report["train"]), ("held out", report["heldout"])):
        lengths = r["approx_prompt_tokens"]
        print(
            f"{label}: {r['prompts']} prompts from {r['conversations']} conversations ({r['tasks']} tasks); "
            + ", ".join(f"{d} {n}" for d, n in r["domains"].items())
            + f"; prompt ≈ {lengths['median']:,} tokens median, {lengths['max']:,} max"
        )
    print(f"wrote {shown(out)}, {heldout_out.name} and {report_out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
