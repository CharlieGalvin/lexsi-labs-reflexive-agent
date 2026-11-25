"""Stage 2b's decision rule: a candidate against SFT v2 on validation, 8 samples each (STAGE2.md).

    python scripts/arms/stage2b_compare.py S2 S2E4            # baseline, candidate (val files)
    python scripts/arms/stage2b_compare.py S2 S2E4 --samples 8 --reps 10000

Per task, each arm's passes are averaged over its samples; the difference (candidate − v2) is
summed over the 30 tasks, so it reads in "tasks of 30", as v2's 17.0 does. Its 95% range is a
paired bootstrap over tasks (``--reps`` resamples, seed 0). The candidate **meets the rule** only
if all three hold, fixed before any Stage 2b result:

- the gain is at least +2.0 (of 30);
- the 95% range is above 0;
- no domain falls by more than 1.5.

Only tasks with exactly ``--samples`` scored conversations in both arms count; any other task is
listed, and the rule isn't checked while one is missing (crashed conversations are re-run first).
Uses the runs' own verdicts, as `samples.py` does. No API calls.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from collections.abc import Iterable
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.metrics import is_broken

DOMAINS = ("retail", "airline", "telecom")
ARMS = ROOT / "results" / "episodes" / "arms"
MIN_GAIN, MAX_DOMAIN_DROP = 2.0, 1.5

Record = dict[str, Any]
Key = tuple[str, str]  # (domain, task id)


def per_task(recs: Iterable[Record], min_sample: int = 0) -> dict[Key, list[bool]]:
    """{(domain, task): [passed, ...]}, one entry per scored conversation (crashes left out), from
    sample ``min_sample`` on."""
    out: dict[Key, list[bool]] = collections.defaultdict(list)
    for r in latest_records(recs):
        if not is_broken(r) and r.get("sample", 0) >= min_sample:
            out[(r["domain"], str(r["task_id"]))].append(bool((r.get("outcome") or {}).get("passed")))
    return dict(out)


def decide(base: dict[Key, list[bool]], cand: dict[Key, list[bool]], samples: int, reps: int) -> dict:
    tasks = sorted(set(base) | set(cand))
    incomplete = [t for t in tasks if len(base.get(t, [])) != samples or len(cand.get(t, [])) != samples]
    both = [t for t in tasks if t not in incomplete]

    def mean(v: list[bool]) -> float:
        return sum(v) / len(v)

    delta = {t: mean(cand[t]) - mean(base[t]) for t in both}
    rng = random.Random(0)
    boots = sorted(sum(delta[rng.choice(both)] for _ in both) for _ in range(reps)) if both else []
    lo, hi = (boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]) if boots else (None, None)
    domains = {
        d: {
            "v2": round(sum(mean(base[t]) for t in both if t[0] == d), 2),
            "candidate": round(sum(mean(cand[t]) for t in both if t[0] == d), 2),
            "change": round(sum(delta[t] for t in both if t[0] == d), 2),
        }
        for d in DOMAINS
        if any(t[0] == d for t in both)
    }
    gain = round(sum(delta.values()), 2)
    checks = {
        f"gain at least +{MIN_GAIN}": gain >= MIN_GAIN,
        "95% range above 0": lo is not None and lo > 0,
        f"no domain down more than {MAX_DOMAIN_DROP}": all(
            v["change"] >= -MAX_DOMAIN_DROP for v in domains.values()
        ),
    }
    return {
        "tasks": len(both),
        "incomplete": [f"{d}/{t}" for d, t in incomplete],
        "v2": round(sum(mean(base[t]) for t in both), 2),
        "candidate": round(sum(mean(cand[t]) for t in both), 2),
        "gain": gain,
        "ci95": [round(lo, 2), round(hi, 2)] if lo is not None and hi is not None else None,
        "domains": domains,
        "checks": checks,
        "meets_rule": not incomplete and all(checks.values()),
    }


def load(arm: str, split: str) -> list[Record]:
    recs: list[Record] = []
    for d in DOMAINS:
        f = ARMS / f"{arm}-{d}-{split}.jsonl"
        if f.exists():
            recs += read_jsonl(f, partial_last_ok=True)
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--split", default="val")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--reps", type=int, default=10000)
    p.add_argument("--min-sample", type=int, default=0, help="only samples from this one on (a check)")
    a = p.parse_args()
    base = per_task(load(a.baseline, a.split), a.min_sample)
    res = decide(base, per_task(load(a.candidate, a.split), a.min_sample), a.samples, a.reps)
    print(f"{a.candidate} against {a.baseline}, {a.split}, {a.samples} samples, {res['tasks']} tasks")
    for d, v in res["domains"].items():
        print(
            f"  {d:<8} {a.baseline} {v['v2']:.2f}  {a.candidate} {v['candidate']:.2f}"
            f"  change {v['change']:+.2f}"
        )
    ci = res["ci95"]
    print(
        f"  total    {a.baseline} {res['v2']:.2f}  {a.candidate} {res['candidate']:.2f}"
        f"  gain {res['gain']:+.2f}" + (f" (95% {ci[0]:+.2f} to {ci[1]:+.2f})" if ci else "")
    )
    if res["incomplete"]:
        print(
            f"⚠ {len(res['incomplete'])} task(s) without {a.samples} scored conversations in both arms, e.g."
        )
        print(f"  {res['incomplete'][:5]}: re-run the crashed ones; the rule isn't checked until then")
    for name, ok in res["checks"].items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"{a.candidate} {'MEETS' if res['meets_rule'] else 'does not meet'} Stage 2b's rule")
    tag = f"-s{a.samples}" + (f"-from{a.min_sample}" if a.min_sample else "") if a.split != "val" else ""
    out = ROOT / "results" / f"stage2b-{a.baseline}-vs-{a.candidate}-{a.split}{tag}.json"
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
