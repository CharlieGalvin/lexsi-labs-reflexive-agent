"""SEL-88: how often the "no yes" rule flags v2's train-task conversations, failures against passes.

    python scripts/labels/intent_rates.py                       # S2-*-trainpool.jsonl → stdout
    python scripts/labels/intent_rates.py --json results/intent-no_yes-train.metrics.json

Train tasks only (`assert_not_test` on every conversation): this is where the rule is built and
tuned. Dev and validation judge the frozen checker once, in SEL-98, never here.

For each domain: the share of failed conversations with at least one finding (we want it high), the
share of passed conversations with one (each is a change the guard would have blocked in a
conversation that went well: we want it low; STAGE2.md's Decision 1 bar is 1 in 10), and the
findings per conversation. Intervals are Wilson 95%. Use ``--show N`` to print N flagged
conversations from passes, the likely wrong blocks, to check the rule by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reflexive.intent import intent_findings
from reflexive.metrics import is_broken
from reflexive.splits import assert_not_test
from reflexive.stats import wilson

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
DOMAINS = ("retail", "airline", "telecom")


def load(path: Path) -> list[dict]:
    recs = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written last line
        if not is_broken(r):
            recs[(str(r["task_id"]), r.get("sample", 0))] = r  # the latest copy of each
    return list(recs.values())


def share(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--arm", default="S2", help="arm file prefix (default S2, SFT v2)")
    p.add_argument("--split", default="trainpool", help="file suffix (default trainpool: train tasks)")
    p.add_argument("--json", type=Path, help="also write the numbers here")
    p.add_argument("--show", type=int, default=0, help="print this many flagged passes to check by hand")
    a = p.parse_args(argv)

    out: dict = {"rule": "no_yes", "arm": a.arm, "split": a.split, "domains": {}}
    shown = 0
    for d in DOMAINS:
        f = a.episodes / f"{a.arm}-{d}-{a.split}.jsonl"
        if not f.exists():
            print(f"⚠ {f.name} not found: left out")
            continue
        recs = load(f)
        rows = {"failed": [0, 0, 0], "passed": [0, 0, 0]}  # flagged conversations, all, findings
        for r in recs:
            assert_not_test(d, str(r["task_id"]))
            found = intent_findings(r)
            key = "passed" if r["outcome"]["passed"] else "failed"
            rows[key][0] += bool(found)
            rows[key][1] += 1
            rows[key][2] += len(found)
            if found and key == "passed" and shown < a.show:
                shown += 1
                print(f"\n--- {d} task {r['task_id']} sample {r.get('sample', 0)}: passed, flagged")
                for x in found:
                    print(f"    turn {x['turn']}: {x['reason']}")
        out["domains"][d] = {
            k: {**share(v[0], v[1]), "findings_per_conversation": v[2] / v[1] if v[1] else None}
            for k, v in rows.items()
        }
        fl, pa = out["domains"][d]["failed"], out["domains"][d]["passed"]
        print(
            f"{d:8s} failures flagged {fl['k']:>3}/{fl['n']:<3}"
            f"  passes flagged {pa['k']:>3}/{pa['n']:<3}"
            f" ({(pa['rate'] or 0):.0%}, 95% CI {pa['ci95'][0]:.0%}–{pa['ci95'][1]:.0%})"
        )
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
