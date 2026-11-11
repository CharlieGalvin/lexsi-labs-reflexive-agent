"""Check the heal rewards on the real data before training (SEL-75). No GPU, no API.

    just grpo-heal-reward-check                  # data/train/grpo-heal.jsonl
    just grpo-heal-reward-check --data data/train/grpo-heal.heldout.jsonl

Two things must hold, or the reward is wrong:

1. **Every teacher turn scores the maximum** (the fixes and the anchor turns): what we train
   towards can't be marked down.
2. **Each failed turn fails its own type's reward**: the student's original turn at a Tool point
   must score 0 on ``tool``, at an Arg point below 1 on ``arg``, at a Loop point 0 on ``loop``.
   Loop is also counted without the "same as the failed turn" rule, which catches any repeat of
   the failed turn: the share caught by the loop rules alone shows they work on their own.

3. **A canned reply can't game it** (Chiara's review): generic replies ("Could you please provide
   more details?") are scored on every prompt. Where the teacher called a tool they must score
   at most ``CANNED_WHERE_CALLED`` (a reply there fails the failure rewards); where it replied, at
   least ``CANNED_GAP`` below the teacher (what a reply says counts); and overall at most
   ``CANNED_MAX`` of the maximum. GRPO compares samples of the same prompt, so where the teacher
   replied the content score is what separates one reply from another.

Exits 1 if any teacher turn scores below the maximum, or a canned reply scores too well.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path
from typing import Any

from reflexive import heal_data, heal_rewards
from reflexive.files import ROOT, read_jsonl, shown

DATA = ROOT / "data" / "train" / "grpo-heal.jsonl"
CANNED = (
    "Could you please provide more details?",
    "Let me look into that for you. One moment, please.",
    "I understand. Is there anything else I can help you with today?",
)
CANNED_WHERE_CALLED = 1.0  # a canned reply's mean where the teacher called a tool
CANNED_GAP = 0.6  # how far below the teacher's maximum it must stay where the teacher replied
CANNED_MAX = 0.75  # its mean over every prompt, as a share of the maximum


def check(rows: list[dict]) -> dict:
    weights = heal_rewards.DEFAULT_WEIGHTS
    top = sum(weights.values())
    below, caught = [], collections.defaultdict(lambda: [0, 0])
    loop_alone = [0, 0]
    for r in rows:
        s = heal_rewards.score_row(r["target"], r)
        if heal_rewards.total(s, weights) < top - 1e-9:
            below.append(
                (r["id"], {k: s[k] for k in heal_rewards.COMPONENTS}, s["arg_kinds"], s["loop_kinds"])
            )
        if r["failure_type"] == heal_data.NONE:
            continue
        f = heal_rewards.score_row(heal_data.failed_text(r["failed"]), r)
        for t in r["failure_types"]:
            caught[t][0] += 1
            caught[t][1] += f[t.lower()] < 1.0
        if "LOOP" in r["failure_types"]:
            loop_alone[0] += 1
            loop_alone[1] += any(k != "failed_turn_again" for k in f["loop_kinds"])
    canned: dict[str, dict[str, Any]] = {}
    for c in CANNED:
        by_kind: dict[str, list[float]] = {"call": [], "text": []}
        for r in rows:
            by_kind[r["kind"]].append(heal_rewards.total(heal_rewards.score_row(c, r), weights))
        everything = by_kind["call"] + by_kind["text"]
        canned[c] = {
            "mean": round(sum(everything) / len(everything), 3) if everything else 0.0,
            "where_teacher_called": round(sum(by_kind["call"]) / len(by_kind["call"]), 3)
            if by_kind["call"]
            else None,
            "where_teacher_replied": round(sum(by_kind["text"]) / len(by_kind["text"]), 3)
            if by_kind["text"]
            else None,
        }
    gamed = [
        c
        for c, v in canned.items()
        if float(v["mean"]) > CANNED_MAX * top
        or (v["where_teacher_called"] is not None and v["where_teacher_called"] > CANNED_WHERE_CALLED)
        or (v["where_teacher_replied"] is not None and v["where_teacher_replied"] > top - CANNED_GAP)
    ]
    return {
        "canned": canned,
        "gamed": gamed,
        "rows": len(rows),
        "teacher_turns_at_max": len(rows) - len(below),
        "below_max": below,
        "failed_turn_caught": {t: {"points": n, "caught": c} for t, (n, c) in sorted(caught.items())},
        "loop_caught_by_loop_rules": {"points": loop_alone[0], "caught": loop_alone[1]},
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA, help=f"default: {shown(DATA)}")
    a = p.parse_args(argv)
    if not a.data.exists():
        print(f"⛔ no {shown(a.data)}: run `just grpo-heal-data` first")
        return 1
    rep = check(read_jsonl(a.data))
    print(
        f"teacher turns at the maximum ({sum(heal_rewards.DEFAULT_WEIGHTS.values()):g}): "
        f"{rep['teacher_turns_at_max']}/{rep['rows']}"
    )
    for t, v in rep["failed_turn_caught"].items():
        print(f"  {t}: the failed turn fails the {t.lower()} reward at {v['caught']}/{v['points']} points")
    la = rep["loop_caught_by_loop_rules"]
    print(
        f"  LOOP by the loop rules alone (not just 'same as the failed turn'): {la['caught']}/{la['points']}"
    )
    top = sum(heal_rewards.DEFAULT_WEIGHTS.values())
    print(
        f"canned replies (must stay ≤ {CANNED_WHERE_CALLED:g} where the teacher called, "
        f"≤ {top - CANNED_GAP:g} where it replied, ≤ {CANNED_MAX * top:g} overall):"
    )
    for c, v in rep["canned"].items():
        print(
            f"  {v['mean']:.2f} overall; {v['where_teacher_called']} where the teacher called, "
            f"{v['where_teacher_replied']} where it replied: {c!r}"
        )
    for rid, parts, arg_kinds, loop_kinds in rep["below_max"][:10]:
        print(f"⛔ {rid}: {parts} {arg_kinds} {loop_kinds}")
    for c in rep["gamed"]:
        print(f"⛔ a canned reply scores too well: {c!r}")
    return 1 if rep["below_max"] or rep["gamed"] else 0


if __name__ == "__main__":
    sys.exit(main())
