"""Decision 1's guard report: wrongly blocked good changes, and what the guard cost (STAGE2.md).

    python scripts/arms/guard_report.py S2G                  # the validation files, S2G-<domain>-val.jsonl
    python scripts/arms/guard_report.py S2G --split train    # any other split's files

**A wrongly blocked good change** (fixed Sun 22 Nov 15:36 UK, before any block was read): a reply
the guard blocked (``action == "blocked"``) with a call that matches one of the task's expected
agent actions by τ²-bench's own rule (`failure_diff.same_call`). τ²-bench expects that change, so
blocking it was wrong. Decision 1's third condition counts **passing** conversations with at least
one; the bar is at most 1 in 10 of the passing conversations, overall, reported per domain too.

Also reported, per domain: blocks per conversation, overridden and held calls, re-asks (the extra
agent turns blocks cost), conversations where a checker error switched the guard off, and judge
errors (``*_judge_error`` findings, which never block). Crashed conversations
(``reflexive.metrics.is_broken``) are left out, as `samples.py` leaves them out. No API calls.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.metrics import is_broken


def _same_call() -> Callable[..., bool]:
    """τ²-bench's matching rule from scripts/arms/failure_diff.py, found from the repo root (so this
    file can also be run from a copy outside the repo, on a pod kept at the frozen tag)."""
    spec = importlib.util.spec_from_file_location(
        "failure_diff", ROOT / "scripts" / "arms" / "failure_diff.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("failure_diff", mod)
    spec.loader.exec_module(mod)
    return mod.same_call


same_call = _same_call()

DOMAINS = ("retail", "airline", "telecom")
ARMS = ROOT / "results" / "episodes" / "arms"
BAR = 0.10

Record = dict[str, Any]
Expected = Callable[[str, str], list[tuple]]  # (domain, task id) -> [(name, arguments, compare_args)]


def tau2_expected(domain: str, task_id: str, _cache: dict[str, Any] = {}) -> list[tuple]:  # noqa: B006
    """The task's expected agent actions, from τ²-bench (as `onpolicy_fixes.py` reads them)."""
    from reflexive.outcome import scorer

    if domain not in _cache:
        _cache[domain] = scorer(domain)
    ec = _cache[domain].task(str(task_id)).evaluation_criteria
    if ec is None:
        return []
    return [(x.name, x.arguments or {}, x.compare_args) for x in ec.actions or [] if x.requestor != "user"]


def guard_of(r: Record) -> dict:
    return (r.get("monitor") or {}).get("actions") or {}


def _args(e: dict) -> dict:
    a = e.get("arguments") or {}
    if isinstance(a, str):
        try:
            a = json.loads(a) if a.strip() else {}
        except json.JSONDecodeError:
            return {"<unparsed>": a}
    return a if isinstance(a, dict) else {"<unparsed>": a}


def wrong_blocks(r: Record, expected: list[tuple]) -> list[dict]:
    """The blocked calls in ``r`` that τ²-bench expects (a good change, blocked)."""
    return [
        e
        for e in guard_of(r).get("blocks") or []
        if e.get("action") == "blocked" and any(same_call(x, (e["tool"], _args(e))) for x in expected)
    ]


def judge_errors(r: Record) -> int:
    n = 0
    for e in guard_of(r).get("blocks") or []:
        rules = [e.get("rule") or "", *((a or {}).get("rule") or "" for a in e.get("also") or [])]
        n += sum(x.endswith("judge_error") for x in rules)
    return n


def report(recs: Iterable[Record], expected: Expected = tau2_expected) -> dict[str, dict[str, Any]]:
    """Per domain and ``all``: conversations, passes, passes with a wrongly blocked good change, and costs."""
    rows: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in recs:
        if is_broken(r):
            continue
        d = r["domain"]
        g = guard_of(r)
        counts = g.get("counts") or {}
        passed = bool((r.get("outcome") or {}).get("passed"))
        wrong = wrong_blocks(r, expected(d, str(r["task_id"]))) if passed else []
        for row in (rows[d], rows["all"]):
            row["conversations"] += 1
            row["passed"] += passed
            row["passed_with_wrong_block"] += bool(wrong)
            row["wrong_blocks"] += len(wrong)
            row["blocked"] += counts.get("blocked", 0)
            row["overridden"] += counts.get("overridden", 0)
            row["held"] += counts.get("held", 0)
            row["reasks"] += len(g.get("reasks") or [])
            row["guard_switched_off"] += bool(g.get("switched_off_by_error"))
            row["judge_errors"] += judge_errors(r)
            row["no_guard_record"] += not g
    out: dict[str, dict[str, Any]] = {}
    for d in [*DOMAINS, "all"]:
        if d not in rows:
            continue
        row: dict[str, Any] = dict(rows[d])
        p = row["passed"]
        row["wrong_block_rate"] = row["passed_with_wrong_block"] / p if p else None
        row["blocks_per_conversation"] = row["blocked"] / row["conversations"]
        out[d] = row
    if "all" in out:
        rate = out["all"]["wrong_block_rate"]
        out["all"]["meets_bar"] = rate is not None and rate <= BAR
    return out


def load(arm: str, split: str) -> list[Record]:
    recs: list[Record] = []
    for d in DOMAINS:
        f = ARMS / f"{arm}-{d}-{split}.jsonl"
        if f.exists():
            recs += latest_records(read_jsonl(f, partial_last_ok=True))
    return recs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm")
    p.add_argument("--split", default="val")
    a = p.parse_args()
    recs = load(a.arm, a.split)
    if not recs:
        raise SystemExit(f"no {a.arm}-<domain>-{a.split}.jsonl files in {ARMS}")
    rep = report(recs)
    for d, row in rep.items():
        rate = row["wrong_block_rate"]
        print(
            f"{d:<8} passes with a wrongly blocked good change"
            f" {row['passed_with_wrong_block']}/{row['passed']}"
            + (f" ({rate:.0%})" if rate is not None else "")
            + f" | blocks {row['blocked']} ({row['blocks_per_conversation']:.2f}/conv), overridden"
            f" {row['overridden']}, held {row['held']}, re-asks {row['reasks']}"
            f" | guard off by checker error {row['guard_switched_off']}, judge errors {row['judge_errors']}"
            + (f" | ⚠ {row['no_guard_record']} without a guard record" if row["no_guard_record"] else "")
        )
    verdict = "meets" if rep["all"]["meets_bar"] else "does NOT meet"
    print(f"condition 3 (at most 1 in 10 passes): {verdict} the bar")
    out = ROOT / "results" / f"guard-report-{a.arm}-{a.split}.json"
    out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"saved {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
