"""SEL-98: the frozen checker against τ²-bench's verdicts, on SFT v2's dev and validation conversations.

    python scripts/labels/checker_vs_verdicts.py                # → stdout
    python scripts/labels/checker_vs_verdicts.py --json results/checker-vs-verdicts-dev-val.metrics.json

Run once, after the freeze (tag ``stage2-checker-frozen-2``), by someone who didn't build the
checker. The frozen checker is the Stage 1 Tool / Arg / Loop classifier plus the made-up-intent
rules: `reflexive.actions.RULES` ("no yes" and any accepted meta-healing rule) and the opted-in
``mismatch`` rule. Each state-changing call is checked against the conversation before it, as the
guard would have checked it live (`reflexive.intent.intent_findings`). A rule that errors (e.g. a
judge call that failed) is counted apart and never flags.

Reported, for each domain and overall, with Wilson 95% intervals:

* **the 37**: failures with no Tool / Arg / Loop finding, and how many the made-up-intent rules
  flag (SEL-98's target: at least half);
* **passes flagged** by the made-up-intent rules (target: no more than 1 in 10), and by the whole
  checker;
* **failures flagged** by the made-up-intent rules and by the whole checker.

Dev and validation only (`assert_not_test` on every conversation), never test. Needs
OPENAI_API_KEY for the rules' judges; every answer is cached under results/cache/, so a re-run is
free and gives the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
CACHE = ROOT / "results" / "cache"
DOMAINS = ("retail", "airline", "telecom")
SPLITS = ("dev", "val")


def load(path: Path) -> list[dict]:
    from reflexive.metrics import is_broken

    recs: dict = {}
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
    from reflexive.stats import wilson

    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [lo, hi]}


def fmt(s: dict) -> str:
    if not s["n"]:
        return "  —"
    return f"{s['k']:>3}/{s['n']:<3} {s['rate']:>4.0%} [{s['ci95'][0]:.0%}–{s['ci95'][1]:.0%}]"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--arm", default="S2", help="arm file prefix (default S2, SFT v2)")
    p.add_argument("--json", type=Path, help="also write the numbers here")
    a = p.parse_args(argv)
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    os.environ.setdefault("REFLEXIVE_META_CACHE", str(CACHE / "meta-rules.jsonl"))

    from reflexive.actions import OPTIONAL_RULES, RULES
    from reflexive.hallucination import classify_record
    from reflexive.intent import intent_findings
    from reflexive.splits import assert_not_test

    rules = [*RULES, OPTIONAL_RULES["mismatch"]]
    keys = (
        "the_37",
        "the_37_flagged",
        "failed",
        "failed_intent",
        "failed_checker",
        "passed",
        "passed_intent",
        "passed_checker",
    )
    counts = {d: dict.fromkeys(keys, 0) for d in (*DOMAINS, "all")}
    by_rule: dict[str, int] = {}
    errors, first_error, missing = 0, "", []
    for d in DOMAINS:
        for s in SPLITS:
            f = a.episodes / f"{a.arm}-{d}-{s}.jsonl"
            if not f.exists():
                missing.append(f.name)
                continue
            for r in load(f):
                assert_not_test(d, str(r["task_id"]))
                tal = bool(classify_record(r).findings)
                every = intent_findings(r, rules=rules)
                live = [x for x in every if x["live"]]
                bad = [x for x in every if not x["live"]]
                errors += len(bad)
                first_error = first_error or next((x["reason"] for x in bad), "")
                for x in live:
                    by_rule[x["rule"]] = by_rule.get(x["rule"], 0) + 1
                intent = bool(live)
                for c in (counts[d], counts["all"]):
                    if r["outcome"]["passed"]:
                        c["passed"] += 1
                        c["passed_intent"] += intent
                        c["passed_checker"] += intent or tal
                    else:
                        c["failed"] += 1
                        c["failed_intent"] += intent
                        c["failed_checker"] += intent or tal
                        if not tal:
                            c["the_37"] += 1
                            c["the_37_flagged"] += intent
    if missing:
        print(f"⚠ not found (left out): {', '.join(missing)}")
    if not counts["all"]["passed"] + counts["all"]["failed"]:
        print(f"no conversations found in {a.episodes}: nothing is written.")
        return 1
    if errors:
        print(f"⚠ {errors} rule errors (judge calls that failed): those changes were NOT flagged.")
        print(f"  first error: {first_error}")
        print("  Usually OPENAI_API_KEY isn't set in this shell. Fix it and run again; nothing is written.")
        return 1

    out: dict = {
        "ticket": "SEL-98",
        "checker": "stage2-checker-frozen-2: Tool/Arg/Loop + "
        + " + ".join(getattr(x, "__name__", str(x)) for x in rules),
        "arm": a.arm,
        "splits": list(SPLITS),
        "targets": {"the_37_flagged": ">= 0.5", "passes_flagged_by_intent": "<= 0.1"},
        "findings_by_rule": by_rule,
        "domains": {},
    }
    print(
        f"{'':8s}  {'the 37 flagged (intent)':<24}  {'passes flagged (intent)':<24}"
        f"  {'passes flagged (checker)':<24}  {'failures flagged (checker)':<24}"
    )
    for d, c in counts.items():
        row = {
            "the_37_flagged": share(c["the_37_flagged"], c["the_37"]),
            "passes_flagged_intent": share(c["passed_intent"], c["passed"]),
            "passes_flagged_checker": share(c["passed_checker"], c["passed"]),
            "failures_flagged_intent": share(c["failed_intent"], c["failed"]),
            "failures_flagged_checker": share(c["failed_checker"], c["failed"]),
        }
        out["domains"][d] = row
        print(
            f"{d:8s}  {fmt(row['the_37_flagged']):<24}  {fmt(row['passes_flagged_intent']):<24}"
            f"  {fmt(row['passes_flagged_checker']):<24}  {fmt(row['failures_flagged_checker']):<24}"
        )
    tot = out["domains"]["all"]
    print(f"\nthe 37: {counts['all']['the_37']} (expected 37). Findings by rule: {by_rule}")
    met37 = (tot["the_37_flagged"]["rate"] or 0) >= 0.5
    metp = (tot["passes_flagged_intent"]["rate"] or 0) <= 0.1
    print(f"target, at least half of the 37 flagged: {'met' if met37 else 'not met'}")
    print(f"target, no more than 1 in 10 passes flagged: {'met' if metp else 'not met'}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
