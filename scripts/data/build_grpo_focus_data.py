"""H3's prompts: H2's prompts, focused on those where SFT v2's samples disagreed about what to change.

    python scripts/data/build_grpo_focus_data.py runs/grpo-decision-aya32b-v2 --total 100

GRPO learns only from differences between the samples of one prompt. In H2's run, 76% of prompt
groups had no variance in the decision reward (`scripts/grpo/diagnose_run.py`), so most of its
decision prompts taught nothing. This keeps the prompts whose decision reward *varied* across
H2's samples, or fell below 1 (a wrong change happened), from H2's ``rewards.jsonl`` (training
data only, never validation), repeats each ``--repeat`` times, and fills up to ``--total`` with
H2's other prompts. Every row's decision column is switched to ``mode: wrong_only``.

Reads ``data/train/grpo-decision.jsonl`` and ``.heldout.jsonl`` (H2's), writes
``data/train/grpo-decision-h3.jsonl``, ``.heldout.jsonl`` and ``.report.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics as st
from pathlib import Path

from reflexive import decision_data as dd
from reflexive.files import ROOT, read_jsonl, sha256, shown

DATA = ROOT / "data" / "train"


def focus_ids(rewards: Path) -> set[str]:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for line in rewards.open():
        s = json.loads(line)
        by[s["id"]].append(float(s.get("decision", 1.0)))
    return {i for i, xs in by.items() if len(xs) > 1 and (st.pstdev(xs) > 0 or min(xs) < 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="H2's run folder, with rewards.jsonl")
    p.add_argument("--source", default="grpo-decision", help="H2's data file stem in data/train")
    p.add_argument("--name", default="grpo-decision-h3")
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    src, held_src = DATA / f"{a.source}.jsonl", DATA / f"{a.source}.heldout.jsonl"
    rows, held = read_jsonl(src), read_jsonl(held_src)
    want = focus_ids(a.run / "rewards.jsonl")
    focus = [r for r in rows if r["id"] in want]
    rest = [r for r in rows if r["id"] not in want]
    rng = random.Random(a.seed)
    rng.shuffle(rest)
    train = [{**r, "id": r["id"] if k == 0 else f"{r['id']}~{k + 1}"} for r in focus for k in range(a.repeat)]
    train += rest[: max(0, a.total - len(train))]
    rng.shuffle(train)
    for r in [*train, *held]:
        r["decision"] = {**r["decision"], "mode": dd.WRONG_ONLY}

    out, held_out = DATA / f"{a.name}.jsonl", DATA / f"{a.name}.heldout.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train))
    held_out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in held))
    report = {
        "from": {"data": shown(src), "sha256": sha256(src), "rewards": shown(a.run / "rewards.jsonl")},
        "focus_prompts": len(focus),
        "focus_by_type": dict(collections.Counter(r["failure_type"] for r in focus).most_common()),
        "repeat": a.repeat,
        "train": len(train),
        "train_by_type": dict(collections.Counter(r["failure_type"] for r in train).most_common()),
        "heldout": len(held),
        "mode": dd.WRONG_ONLY,
        "files": {out.name: sha256(out), held_out.name: sha256(held_out)},
    }
    (DATA / f"{a.name}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(focus)} focus prompts (decision varied or fell below 1 in H2): {report['focus_by_type']}")
    print(f"train {len(train)}: {report['train_by_type']}; heldout {len(held)}; mode wrong_only")
    print(f"wrote {shown(out)}, {held_out.name} and {a.name}.report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
