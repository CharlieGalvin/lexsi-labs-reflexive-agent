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

Exits 1 if any teacher turn scores below the maximum.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from reflexive import heal_data, heal_rewards
from reflexive.files import ROOT, read_jsonl, shown

DATA = ROOT / "data" / "train" / "grpo-heal.jsonl"


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
    return {
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
    for rid, parts, arg_kinds, loop_kinds in rep["below_max"][:10]:
        print(f"⛔ {rid}: {parts} {arg_kinds} {loop_kinds}")
    return 1 if rep["below_max"] else 0


if __name__ == "__main__":
    sys.exit(main())
