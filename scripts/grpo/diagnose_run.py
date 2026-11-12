"""Where a heal GRPO run's learning signal went: per-step trainer logs and per-prompt reward groups.

    python scripts/grpo/diagnose_run.py runs/grpo-decision-aya32b-v2

Reads ``log.jsonl`` (TRL's per-step logs) and ``rewards.jsonl`` (every scored sample) and prints:

* **Zero-variance groups:** GRPO learns only from differences between the samples of one prompt.
  A group whose 8 samples score the same (std 0) contributes no gradient. Shown for the total
  reward and for each component, overall and by failure type.
* **Which components carry the signal:** each component's share of the within-group variance.
* **Truncation, KL and drift per step:** clipped completions (hit ``max_completion_length``), the
  KL to the starting adapter (how far it moved), and the reply share against the teacher's.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics as st
from pathlib import Path

COMPONENTS = ("tool", "arg", "loop", "fix", "guard", "decision")


def groups(samples: list[dict]) -> dict[tuple, list[dict]]:
    """Samples by (step, prompt id): one GRPO group each."""
    out: dict[tuple, list[dict]] = collections.defaultdict(list)
    for s in samples:
        out[(s.get("step"), s.get("id"))].append(s)
    return {k: v for k, v in out.items() if len(v) > 1}


def sd(xs: list[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="the run folder, e.g. runs/grpo-decision-aya32b-v2")
    a = p.parse_args()
    samples = [json.loads(line) for line in (a.run / "rewards.jsonl").open()]
    g = groups(samples)
    comps = [c for c in COMPONENTS if c in samples[0]]
    per = len(samples) / max(len(g), 1)
    print(f"{len(samples)} samples in {len(g)} groups (prompt x step), {per:.1f} per group\n")

    print("Groups with NO variance (all samples score the same: no gradient from that prompt):")
    print(f"  total reward: {sum(sd([s['reward'] for s in v]) < 1e-9 for v in g.values()) / len(g):.0%}")
    for c in comps:
        z = sum(sd([s[c] for s in v]) < 1e-9 for v in g.values()) / len(g)
        mean = st.fmean(s[c] for s in samples)
        print(f"  {c:<9} {z:4.0%} zero-variance   mean {mean:.2f}")

    print("\nBy failure type (zero-variance share of the total reward; mean decision; groups):")
    by: dict[str, list] = collections.defaultdict(list)
    for v in g.values():
        by[v[0].get("failure_type") or "?"].append(v)
    for ft, vs in sorted(by.items()):
        z = sum(sd([s["reward"] for s in v]) < 1e-9 for v in vs) / len(vs)
        dz = sum(sd([s.get("decision", 1) for s in v]) < 1e-9 for v in vs) / len(vs)
        dm = st.fmean(s.get("decision", 1) for v in vs for s in v)
        print(f"  {ft:<13} total {z:4.0%} zero  decision {dz:4.0%} zero, mean {dm:.2f}  ({len(vs)} groups)")

    print("\nWhich components carry the within-group differences (share of summed variance):")
    var = {c: sum(st.pvariance([s[c] for s in v]) for v in g.values()) for c in comps}
    tot = sum(var.values()) or 1
    for c in comps:
        print(f"  {c:<9} {var[c] / tot:5.0%}")

    log_path = a.run / "log.jsonl"
    if log_path.exists():
        logs = [json.loads(line) for line in log_path.open() if '"loss"' in line or "reward" in line]
        keys = (
            "frac_reward_zero_std",
            "reward_std",
            "kl",
            "completions/clipped_ratio",
            "heal/reply_share",
            "rewards/decision/std",
        )
        print("\nPer step, averaged over blocks of 10 steps:")
        print("  steps    " + "  ".join(f"{k.split('/')[-1][:14]:>14}" for k in keys))
        for i in range(0, len(logs), 10):
            b = logs[i : i + 10]
            vals = []
            for k in keys:
                xs = [float(x[k]) for x in b if k in x]
                vals.append(f"{st.fmean(xs):14.3f}" if xs else f"{'-':>14}")
            print(f"  {b[0]['step']:3d}-{b[-1]['step']:<3d}  " + "  ".join(vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
