"""All the headline metrics for an arm on τ²-bench, from its rollout files (SEL-38).

    just arm-eval results/episodes/arms/A-*-test.jsonl
    just arm-eval results/episodes/arms/S-*-dev.jsonl --success saved          # no re-scoring
    just arm-eval <files> --tool-selection openai:gpt-4.1-mini                   # + the LLM judge
    python scripts/arms/evaluate.py <files> --json results/eval-A-test.json

For each file (one arm, one domain, one split) and for all of them together: success from the
AuditKIT oracle (`reflexive.oracle`: τ²-bench's scoring, re-run; ``--success saved`` reads the
run's own verdict instead, free), AuditKIT's agent scorers (``tool_call_validity``,
``redundant_tool_calls``, ``agent_loop_detection``; ``tool_selection`` with
``--tool-selection MODEL``, about one judge call per tool call), and TTC: tokens, cost and
seconds per conversation, and successes per million tokens and per dollar. See
`reflexive.evaluate` for each definition.

Reads results only, so it runs on any split, test included. Crashed conversations are counted
and left out (re-run them first); a re-run's record replaces the crashed one.

**Judges:** the oracle's LLM-judged checks (retail, airline) follow ``--judges`` as in
``oracle-score`` (auto: on with ``OPENAI_API_KEY``). With them off, some verdicts are
``unknown`` and the success rate isn't given.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.evaluate import evaluate, oracle_verdict, saved_verdict
from reflexive.files import BadJsonLine, read_jsonl, shown

ROW = (
    "{name:<22} {succ:<26} valid {tool_call_validity}  redundant {redundant_tool_calls}  "
    "loop-free {agent_loop_detection}{sel}  tokens/conv {total:<9} ${cost_usd_mean}  {seconds_median}s"
)


def line(name: str, s: dict[str, Any]) -> str:
    if s["success_bounds"]:
        lo, hi = s["success_bounds"]
        succ = f"{lo:.0%}–{hi:.0%} ({s['verdicts']['unknown']} unknown)"
    elif s["success_rate"] is None:
        succ = "n/a"
    else:
        lo, hi = s["success_wilson95"]
        k, n = s["verdicts"]["success"], s["verdicts"]["success"] + s["verdicts"]["failure"]
        succ = f"{k}/{n} = {s['success_rate']:.0%} ({lo:.0%}–{hi:.0%})"
    total = s["total_tokens_mean"] if s["user_tokens_recorded"] else f"{s['agent_tokens_mean']}*"
    sel = f"  selection {s['tool_selection']}" if "tool_selection" in s else ""
    return ROW.format(name=name, succ=succ, sel=sel, total=total, **s)


def split_of(recs: list[dict]) -> str:
    found = {splits.split_of(r["domain"], str(r["task_id"])) for r in recs if "domain" in r}
    return found.pop() if len(found) == 1 else "mixed"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="rollout files (results/episodes/arms/*.jsonl)")
    p.add_argument("--success", choices=["oracle", "saved"], default="oracle")
    p.add_argument("--judges", choices=["auto", "on", "off"], default="auto", help="the oracle's judges")
    p.add_argument("--tool-selection", metavar="MODEL", help="also run AuditKIT's tool_selection judge")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--json", type=Path, help="write every summary and conversation here")
    a = p.parse_args()

    from loguru import logger

    logger.remove()  # τ²-bench logs every step
    if a.success == "oracle":
        from reflexive.oracle import Tau2Oracle

        key = bool(os.environ.get("OPENAI_API_KEY"))
        if a.judges == "on" and not key:
            print("⛔ --judges on needs OPENAI_API_KEY")
            return 1
        oracle = Tau2Oracle(judges={"on": True, "off": False}.get(a.judges, key))
        success, how = oracle_verdict(oracle), f"oracle {oracle.identity()['version']}"
    else:
        success, how = saved_verdict, "the runs' saved verdicts"
    judge = None
    if a.tool_selection:
        from auditkit.metrics.agent import ToolSelectionJudge

        judge = ToolSelectionJudge(judge_model=a.tool_selection)
    print(
        f"success: {how}; scorers: AuditKIT agent metrics"
        + (f" + tool_selection ({a.tool_selection})" if judge else "")
    )

    report: dict[str, Any] = {"success": how, "tool_selection": a.tool_selection, "files": {}}
    all_rows: list[dict] = []
    crashed = 0
    for f in a.files:
        try:
            recs = read_jsonl(f, partial_last_ok=True)
        except (BadJsonLine, FileNotFoundError) as e:
            print(f"⛔ {e}")
            return 1
        rows, s = evaluate(recs, success, judge, a.workers)
        s["split"] = split_of(recs)
        print(line(f.stem, s))
        report["files"][shown(f)] = {"summary": s, "conversations": rows}
        all_rows += rows
        crashed += s["crashed_not_scored"]
    if len(a.files) > 1:
        from reflexive.evaluate import summarise

        overall = summarise(all_rows, crashed)
        print(line("overall", overall))
        report["overall"] = overall
    if crashed:
        print(f"⚠ {crashed} crashed conversations left out: re-run them for a complete result")
    if not all(v["summary"]["user_tokens_recorded"] for v in report["files"].values()):
        print("* agent tokens only: these records predate the customer's token count (its cost is included)")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1) + "\n")
        print(f"→ {shown(a.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
