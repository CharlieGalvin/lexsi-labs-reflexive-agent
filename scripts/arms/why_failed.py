"""Why conversations failed, check by check: unmet device assertions and missed actions (SEL-75).

    just arm-why S2 H --domain telecom                  # dev (default): v2 against H
    just arm-why A --domain telecom --split train --dir data/golden/hub-v1.1/raw/episodes/arms

A saved rollout keeps only one score per check (``ENV_ASSERTION: 0``). This re-scores each failed
conversation that ended normally with τ²-bench's own evaluator (`reflexive.outcome.score`, judges
off: telecom has none) to get the detail: **which device assertions weren't met** (telecom: the
customer's phone still has mobile data off, the APN broken, roaming off …) and **which expected
actions were missed**, the agent's own and the customer's (in telecom the agent must get the
customer to do things on the phone). With two arms, it also lists, for each conversation the first
passed and the second failed (same task and sample), what the second left undone.

Needs τ²-bench (the pod's environment); no GPU, no API calls.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl, shown

NORMAL = {"agent_stop", "user_stop"}
DIR = ROOT / "results" / "episodes" / "arms"

Record = dict[str, Any]


def load(folder: Path, arm: str, domain: str, split: str) -> dict[tuple[str, int], Record]:
    path = folder / f"{arm}-{domain}-{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"⛔ no {shown(path)}")
    return {
        (str(r["task_id"]), int(r.get("sample", 0))): r
        for r in latest_records(read_jsonl(path, partial_last_ok=True))
    }


def _call(name: str, args: dict[str, Any]) -> str:
    inner = ", ".join(f"{k}={json.dumps(v)}" for k, v in sorted(args.items()))
    return f"{name}({inner})"


def _quiet() -> None:
    """τ²-bench logs every tool it replays and prints the phone's status bar: keep only ours."""
    try:
        from loguru import logger

        logger.remove()
    except ImportError:
        pass


def detail(domain: str, rec: Record) -> dict[str, Any]:
    """Unmet assertions and missed actions of one failed conversation, re-scored."""
    from reflexive.outcome import score

    _quiet()
    with contextlib.redirect_stdout(io.StringIO()):
        out = score(domain, str(rec["task_id"]), rec["messages"], str(rec.get("termination")), judges=False)
    info = out.reward_info
    unmet = [
        _call(a.env_assertion.func_name, a.env_assertion.arguments)
        + ("" if a.env_assertion.assert_value else " is False")
        for a in (getattr(info, "env_assertions", None) or [])
        if not a.met
    ]
    missed = [
        f"{a.action.requestor}: {_call(a.action.name, a.action.arguments)}"
        for a in (getattr(info, "action_checks", None) or [])
        if not a.action_match
    ]
    db = getattr(getattr(info, "db_check", None), "db_match", None)
    return {"unmet": unmet, "missed": missed, "db_match": db, "checks": out.checks}


def customer_actions(rec: Record) -> int:
    """How many tool calls the customer made on their phone (telecom), as the agent asked."""
    return sum(len(m.get("tool_calls") or []) for m in rec.get("messages") or [] if m.get("role") == "user")


def summarise(domain: str, recs: dict[tuple[str, int], Record]) -> dict[str, Any]:
    endings: collections.Counter[str] = collections.Counter()
    unmet: collections.Counter[str] = collections.Counter()
    missed: collections.Counter[str] = collections.Counter()
    for r in recs.values():
        if (r.get("outcome") or {}).get("passed"):
            endings["passed"] += 1
        elif r.get("termination") not in NORMAL:
            endings[f"ended: {r.get('termination')}"] += 1
        else:
            endings["failed, ended normally"] += 1
            d = detail(domain, r)
            unmet.update(d["unmet"])
            missed.update(d["missed"])
    return {"endings": endings, "unmet": unmet, "missed": missed, "n": len(recs)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+", help="one arm, or two to compare (the first is the reference)")
    p.add_argument("--domain", default="telecom", choices=["retail", "airline", "telecom"])
    p.add_argument("--split", default="dev")
    p.add_argument("--dir", type=Path, default=DIR, help=f"where the rollouts are (default {shown(DIR)})")
    p.add_argument("--top", type=int, default=12)
    a = p.parse_args(argv)

    runs = {arm: load(a.dir, arm, a.domain, a.split) for arm in a.arms}
    for arm, recs in runs.items():
        s = summarise(a.domain, recs)
        print(f"\n# {arm} ({a.domain} {a.split}): {s['n']} conversations; {dict(s['endings'])}")
        mean = sum(customer_actions(r) for r in recs.values()) / max(len(recs), 1)
        print(f"  customer actions per conversation: {mean:.1f}")
        print("  device assertions not met (in failed conversations that ended normally):")
        for k, n in s["unmet"].most_common(a.top):
            print(f"    {n:3d}  {k}")
        print("  expected actions missed:")
        for k, n in s["missed"].most_common(a.top):
            print(f"    {n:3d}  {k}")

    if len(a.arms) >= 2:
        ref, other = a.arms[0], a.arms[1]
        r1, r2 = runs[ref], runs[other]
        lost = [
            k
            for k in sorted(r1.keys() & r2.keys())
            if r1[k]["outcome"].get("passed") and not r2[k]["outcome"].get("passed")
        ]
        print(f"\n# {len(lost)} conversations {ref} passed and {other} failed (same task and sample)")
        for task, sample in lost:
            rec = r2[(task, sample)]
            print(
                f"\n## {task} (sample {sample}): {other} {rec.get('termination')}, "
                f"{len(rec['messages'])} messages ({ref} {len(r1[(task, sample)]['messages'])}); "
                f"customer actions {customer_actions(rec)} ({ref} {customer_actions(r1[(task, sample)])})"
            )
            if rec.get("termination") not in NORMAL:
                print("   ended abnormally: no assertion detail")
                continue
            d = detail(a.domain, rec)
            for u in d["unmet"]:
                print(f"   not met: {u}")
            for m in d["missed"]:
                print(f"   missed:  {m}")
            if not d["unmet"] and not d["missed"]:
                print(f"   checks: {d['checks']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
