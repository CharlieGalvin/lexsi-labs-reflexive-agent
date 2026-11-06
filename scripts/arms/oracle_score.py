"""Score saved rollouts with the AuditKIT oracle: success rate from verified state (SEL-46).

    just oracle-score results/episodes/arms/A-retail-dev.jsonl results/episodes/arms/A-airline-dev.jsonl
    just oracle-score results/episodes/arms/A-telecom-train.jsonl --only dev
    just oracle-score <files> --judges off      # deterministic checks only: free, but some → unknown
    python scripts/arms/oracle_score.py <files> --workers 8 --json results/oracle.json

Each conversation is re-scored by ``reflexive.oracle.Tau2Oracle`` (τ²-bench's evaluator, inside
AuditKIT), never read from the record. Prints the verdicts (success / failure / unknown /
error), the success rate over scored conversations with its 95% Wilson range, and every
conversation where the oracle disagrees with the verdict the run saved. With ``unknown``
verdicts (judges off), the success rate is given as bounds: unknowns counted as failures,
then as successes.

**Judges** (``--judges auto``, the default, as in ``rollout.py``: on when ``OPENAI_API_KEY`` is
set): retail's and airline's LLM-judged checks call OpenAI, about a cent per conversation.
``--judges on`` without the key is refused. With judges off, a conversation that passes its
deterministic checks is ``unknown``, not ``success``.

Conversations are scored ``--workers`` at a time (default 4; the judges are API calls). A
half-written last line (a rollout still running, or stopped) is skipped. A record that can't be
made into an episode is an ``error`` verdict, not a traceback. Each conversation is keyed by
arm, domain, task and sample: the same one in two files is counted once overall, with a warning.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import BadJsonLine, read_jsonl, shown
from reflexive.oracle import Tau2Oracle, case_for, episode_from_record
from reflexive.stats import wilson


def key(r: dict[str, Any]) -> str:
    """One conversation: arm, domain, task and sample (``A retail:12 #0``)."""
    return f"{r.get('arm', '?')} {r.get('domain')}:{r.get('task_id')} #{r.get('sample', 0)}"


def score_one(r: dict[str, Any], oracle: Tau2Oracle) -> tuple[dict[str, Any], Any, Any]:
    """(record, verdict, the record's own saved verdict). A malformed record is an ``error``."""
    from auditkit.agent_eval.outcome import ERROR, Outcome, resolve_outcome

    try:
        ep = episode_from_record(r)
    except Exception as e:  # noqa: BLE001 - one bad record shouldn't stop the run
        why = f"the record can't be read as a conversation ({type(e).__name__}: {str(e)[:200]})"
        return r, Outcome(ERROR, reason=why, source=oracle.identity()), None
    return r, resolve_outcome(ep, case_for(r, oracle)), ep.source_verdict


def score_all(
    recs: list[dict[str, Any]], oracle: Tau2Oracle, workers: int = 1
) -> list[tuple[dict[str, Any], Any, Any]]:
    """``score_one`` for each record, ``workers`` at a time, in the records' order."""
    with ThreadPoolExecutor(max(1, workers)) as pool:
        return list(pool.map(lambda r: score_one(r, oracle), recs))


def in_split(r: dict[str, Any], only: str) -> bool:
    """Whether a record's task is in the ``only`` split; a task in no split (or no task) isn't."""
    try:
        return splits.split_of(r["domain"], str(r["task_id"])) == only
    except KeyError:
        return False


def summary(scored: list[tuple[dict[str, Any], Any, Any]]) -> dict[str, Any]:
    """Verdict counts, success rate over success + failure, and disagreements with the run."""
    counts = collections.Counter(o.verdict for _, o, _ in scored)
    k, n, u = counts["success"], counts["success"] + counts["failure"], counts["unknown"]
    lo, hi = wilson(k, n)
    disagree = [
        {"task": key(r), "oracle": o.verdict, "saved": saved, "reason": o.reason}
        for r, o, saved in scored
        if saved and o.verdict in ("success", "failure") and o.verdict != saved
    ]
    return {
        "verdicts": {v: counts[v] for v in ("success", "failure", "unknown", "error")},
        "success_rate": round(k / n, 4) if n and not u else None,
        "success_rate_wilson95": [round(lo, 4), round(hi, 4)] if n and not u else None,
        # with unknowns: the rate if every unknown failed, and if every unknown succeeded
        "success_rate_bounds": [round(k / (n + u), 4), round((k + u) / (n + u), 4)] if u else None,
        "scored": n,
        "disagreements_with_saved": disagree,
    }


def rate_text(s: dict[str, Any]) -> str:
    """The success rate as one line: a rate with its range, bounds with unknowns, or n/a."""
    v = s["verdicts"]
    if s["success_rate_bounds"]:
        lo, hi = s["success_rate_bounds"]
        return f"between {lo:.1%} and {hi:.1%} ({v['unknown']} unknown: judges off)"
    if s["success_rate"] is None:
        return "n/a (nothing scored)"
    lo, hi = s["success_rate_wilson95"]
    return f"{v['success']}/{s['scored']} = {s['success_rate']:.1%} ({lo:.1%}–{hi:.1%})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="rollout files (results/episodes/arms/*.jsonl)")
    p.add_argument("--only", choices=["train", "dev"], help="keep only this frozen split's tasks")
    p.add_argument(
        "--judges", choices=["auto", "on", "off"], default="auto", help="auto: on with OPENAI_API_KEY"
    )
    p.add_argument("--workers", type=int, default=4, help="conversations scored at once (default 4)")
    p.add_argument("--json", type=Path, help="also write the summary per file, and overall, here")
    a = p.parse_args()
    key_set = bool(os.environ.get("OPENAI_API_KEY"))
    if a.judges == "on" and not key_set:
        print(
            "⛔ --judges on needs OPENAI_API_KEY (retail's and airline's judges call OpenAI): "
            "set it, or pass --judges off"
        )
        return 1
    if len({f.resolve() for f in a.files}) < len(a.files):
        print("⛔ the same file is given twice: each conversation would count twice overall")
        return 1
    from loguru import logger

    logger.remove()  # τ²-bench logs every step
    oracle = Tau2Oracle(judges={"on": True, "off": False}.get(a.judges, key_set))
    print(
        f"oracle: {oracle.identity()['version']}"
        + (" (auto: no OPENAI_API_KEY)" if a.judges == "auto" and not key_set else "")
    )
    report: dict[str, Any] = {"oracle": oracle.identity(), "files": {}}
    everything, seen = [], set()
    for f in a.files:
        try:
            recs = read_jsonl(f, partial_last_ok=True)
        except (BadJsonLine, FileNotFoundError) as e:
            print(f"⛔ {e}")
            return 1
        if a.only:
            unsplit = [r for r in recs if not any(in_split(r, sp) for sp in splits.SPLITS)]
            if unsplit:
                print(
                    f"⚠ {shown(f)}: {len(unsplit)} records in no frozen split "
                    f"(e.g. {key(unsplit[0])}): left out"
                )
            recs = [r for r in recs if in_split(r, a.only)]
        scored = score_all(recs, oracle, a.workers)
        repeats = [x for x in scored if key(x[0]) in seen]
        if repeats:
            print(
                f"⚠ {shown(f)}: {len(repeats)} conversations already in an earlier file "
                f"(e.g. {key(repeats[0][0])}): counted once overall"
            )
        everything += [x for x in scored if key(x[0]) not in seen]
        seen |= {key(r) for r in recs}
        if not recs:
            print(f"⚠ {shown(f)}: no records{f' in the {a.only} split' if a.only else ''}")
        s = summary(scored)
        report["files"][shown(f)] = s
        v = s["verdicts"]
        print(
            f"{shown(f)}{f' [{a.only}]' if a.only else ''}: {v['success']} success, {v['failure']} failure, "
            f"{v['unknown']} unknown, {v['error']} error → success rate {rate_text(s)}"
        )
        for d in s["disagreements_with_saved"]:
            print(f"   ⚠ {d['task']}: oracle says {d['oracle']}, the run saved {d['saved']} ({d['reason']})")
    if len(a.files) > 1:
        s = summary(everything)
        report["overall"] = s
        v = s["verdicts"]
        print(f"overall: success rate {rate_text(s)}, {v['unknown']} unknown, {v['error']} error")
    if a.json:
        a.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {shown(a.json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
