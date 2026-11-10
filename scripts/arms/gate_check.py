"""Run the agreed DevGate (configs/gate.yaml) on saved dev runs: is the candidate a valid heal?

    just gate-check P S                    # P = current model, S = candidate; sample 0 of each
    just gate-check P S --sample 1         # another sample (decide which one BEFORE looking)
    just gate-check P S2 --record          # an official gate decision: counts as a heal cycle

Reads ``results/episodes/arms/<arm>-<domain>-dev.jsonl`` for both arms (the last record per
task and sample, so a re-run replaces its crash), keeps one sample per task as the gate
requires, and re-scores everything with the AuditKIT oracle (judges on). Prints the decision,
the check that made it, and the numbers behind it, and writes them to
``results/gate-<baseline>-vs-<candidate>-s<sample>.json``.

Without ``--record`` it's a check: the decision isn't added to the gate's history
(``results/gate_history.jsonl``), so it doesn't use up one of the ``max_cycles`` heal cycles.
It still reads that history, so once the budget is spent a check says so too.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from reflexive.files import ROOT, latest_records, read_jsonl, shown
from reflexive.gate import DevGate

HISTORY = ROOT / "results" / "gate_history.jsonl"


def dev_run(arm: str, sample: int) -> list[dict]:
    recs = [
        r
        for f in sorted(glob.glob(str(ROOT / "results" / "episodes" / "arms" / f"{arm}-*-dev.jsonl")))
        for r in read_jsonl(Path(f), partial_last_ok=True)
        if r.get("sample", 0) == sample
    ]
    return latest_records(recs)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("baseline", help="the current model's arm key, e.g. P")
    p.add_argument("candidate", help="the candidate's arm key, e.g. S")
    p.add_argument("--sample", type=int, default=0, help="which sample per task (default 0)")
    p.add_argument("--record", action="store_true", help="add the decision to the gate's history")
    a = p.parse_args()

    try:
        from loguru import logger

        logger.remove()  # τ²-bench logs every step
    except ImportError:  # no τ²-bench here (CI): nothing to silence
        pass
    base, cand = dev_run(a.baseline, a.sample), dev_run(a.candidate, a.sample)
    print(
        f"{a.baseline}: {len(base)} conversations, {a.candidate}: {len(cand)} (sample {a.sample}); scoring..."
    )
    gate = DevGate.load(history_file=HISTORY)  # the budget counts every recorded decision
    if not a.record:
        gate.history_file = None  # a check reads the history but never adds to it
    print(f"heal cycles gated so far: {gate.cycles_used} of {gate.max_cycles}")
    d = gate.decide(base, cand)
    note = "" if a.record else "  [check only: not recorded]"
    print(f"\nGATE: {d.verdict.upper()} (decided by: {d.check}){note}")
    print(f"pass rate: {a.baseline} {d.baseline_rate:.1%} -> {a.candidate} {d.candidate_rate:.1%}")
    print(f"reason: {d.reason}")
    keys = ("passed", "only_baseline_passed", "only_candidate_passed", "net_gain", "mcnemar_p", "domains_net")
    for k in keys:
        if k in d.details:
            print(f"  {k}: {d.details[k]}")
    if f := d.details.get("findings_per_conversation"):
        print(f"  findings per conversation: {f['baseline']['total']} -> {f['candidate']['total']}")
    out = HISTORY.parent / f"gate-{a.baseline}-vs-{a.candidate}-s{a.sample}.json"
    out.write_text(
        json.dumps(
            {
                "baseline": a.baseline,
                "candidate": a.candidate,
                "sample": a.sample,
                "recorded": a.record,
                "verdict": d.verdict,
                "check": d.check,
                "reason": d.reason,
                "details": d.details,
            },
            indent=1,
            default=str,
        )
        + "\n"
    )
    print(f"→ {shown(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
