"""Score saved rollouts with the AuditKIT oracle: success rate from verified state (SEL-46).

    just oracle-score results/episodes/arms/A-retail-dev.jsonl results/episodes/arms/A-airline-dev.jsonl
    just oracle-score results/episodes/arms/A-telecom-train.jsonl --only dev
    just oracle-score <files> --judges off      # deterministic checks only: free, but some → unknown

Each conversation is re-scored by ``reflexive.oracle.Tau2Oracle`` (τ²-bench's evaluator, inside
AuditKIT), never read from the record. Prints the verdicts (success / failure / unknown /
error), the success rate over scored conversations with its 95% Wilson range, and every
conversation where the oracle disagrees with the verdict the run saved.

**Judges** (default on): retail's and airline's LLM-judged checks call OpenAI (needs
``OPENAI_API_KEY``, about a cent per conversation). With ``--judges off`` a conversation that
passes its deterministic checks is ``unknown``, not ``success``.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import BadJsonLine, read_jsonl, shown
from reflexive.oracle import Tau2Oracle, case_for, episode_from_record
from reflexive.stats import wilson


def score_all(recs: list[dict[str, Any]], oracle: Tau2Oracle) -> list[tuple[dict[str, Any], Any, Any]]:
    """(record, verdict, the record's own saved verdict) for each record."""
    from auditkit.agent_eval.outcome import resolve_outcome

    out = []
    for r in recs:
        ep = episode_from_record(r)
        out.append((r, resolve_outcome(ep, case_for(r, oracle)), ep.source_verdict))
    return out


def summary(scored: list[tuple[dict[str, Any], Any, Any]]) -> dict[str, Any]:
    """Verdict counts, success rate over success + failure, and disagreements with the run."""
    counts = collections.Counter(o.verdict for _, o, _ in scored)
    k, n = counts["success"], counts["success"] + counts["failure"]
    lo, hi = wilson(k, n)
    disagree = [
        {"task": f"{r['domain']}:{r['task_id']}", "oracle": o.verdict, "saved": saved, "reason": o.reason}
        for r, o, saved in scored
        if saved and o.verdict in ("success", "failure") and o.verdict != saved
    ]
    return {
        "verdicts": {v: counts[v] for v in ("success", "failure", "unknown", "error")},
        "success_rate": round(k / n, 4) if n else None,
        "success_rate_wilson95": [round(lo, 4), round(hi, 4)],
        "scored": n,
        "disagreements_with_saved": disagree,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="rollout files (results/episodes/arms/*.jsonl)")
    p.add_argument("--only", choices=["train", "dev"], help="keep only this frozen split's tasks")
    p.add_argument("--judges", choices=["on", "off"], default="on")
    p.add_argument("--json", type=Path, help="also write the summary per file, and overall, here")
    a = p.parse_args()
    from loguru import logger

    logger.remove()  # τ²-bench logs every step
    oracle = Tau2Oracle(judges=a.judges == "on")
    print(f"oracle: {oracle.identity()['version']}")
    report: dict[str, Any] = {"oracle": oracle.identity(), "files": {}}
    everything = []
    for f in a.files:
        try:
            recs = read_jsonl(f)
        except (BadJsonLine, FileNotFoundError) as e:
            print(f"⛔ {e}")
            return 1
        if a.only:
            recs = [r for r in recs if splits.split_of(r["domain"], str(r["task_id"])) == a.only]
        scored = score_all(recs, oracle)
        everything += scored
        s = summary(scored)
        report["files"][shown(f)] = s
        v, (lo, hi) = s["verdicts"], s["success_rate_wilson95"]
        rate = f"{s['success_rate']:.1%} ({lo:.1%}–{hi:.1%})" if s["scored"] else "n/a"
        print(
            f"{shown(f)}{f' [{a.only}]' if a.only else ''}: {v['success']} success, {v['failure']} failure, "
            f"{v['unknown']} unknown, {v['error']} error → success rate {rate}"
        )
        for d in s["disagreements_with_saved"]:
            print(f"   ⚠ {d['task']}: oracle says {d['oracle']}, the run saved {d['saved']} ({d['reason']})")
    if len(a.files) > 1:
        s = summary(everything)
        report["overall"] = s
        lo, hi = s["success_rate_wilson95"]
        v = s["verdicts"]
        print(
            f"overall: {v['success']}/{s['scored']} = {s['success_rate']:.1%} ({lo:.1%}–{hi:.1%}), "
            f"{v['unknown']} unknown, {v['error']} error"
        )
    if a.json:
        a.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {shown(a.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
