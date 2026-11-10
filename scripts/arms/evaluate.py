"""All the headline metrics for an arm on τ²-bench, from its rollout files (SEL-38).

    just arm-eval results/episodes/arms/A-*-test.jsonl
    just arm-eval results/episodes/arms/S-*-dev.jsonl --success saved          # no re-scoring
    just arm-eval <files> --tool-selection openai:gpt-4.1-mini                   # + the LLM judge
    python scripts/arms/evaluate.py <files> --json results/eval-A-test.json

For each file (one arm, one domain, one split) and, when the files hold one arm on one split
(e.g. its three domains), for all of them together: success from the
AuditKIT oracle (`reflexive.oracle`: τ²-bench's scoring, re-run; ``--success saved`` reads the
run's own verdict instead, free), AuditKIT's agent scorers (``tool_call_validity``,
``redundant_tool_calls``, ``agent_loop_detection``; ``tool_selection`` with
``--tool-selection MODEL``, about one judge call per tool call), and TTC: tokens, cost and
seconds per conversation, and successes per million tokens and per dollar. See
`reflexive.evaluate` for each definition.

Reads results only, so it runs on any split, test included. Crashed conversations are counted
and left out (re-run them first); a re-run's record replaces the crashed one, also when the
re-run is in a later file. No τ²-bench needed with ``--success saved``: the tool schemas come
from ``configs/tool_schemas.json``.

The success range resamples tasks (a task's samples together), and each sample's rate is shown
too. An ``unknown`` or ``error`` verdict gives a range of rates instead of one rate, and no
successes per token or per dollar. Successes per dollar count API spend only (crashed
conversations' too): a self-hosted student is $0, so compare arms per token. Tokens are agent
plus customer, or agent only (marked ``*``) when some records predate the customer's count.

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
from reflexive.evaluate import evaluate, oracle_verdict, saved_verdict, summarise, warm
from reflexive.files import BadJsonLine, latest_records, read_jsonl, shown
from reflexive.metrics import is_broken, record_cost

ROW = (
    "{name:<22} {succ:<26} valid {tool_call_validity}  redundant {redundant_tool_calls}  "
    "loop-free {agent_loop_detection}{sel}  tokens/conv {total:<9} ${cost_usd_mean}  {seconds_median}s"
)


def line(name: str, s: dict[str, Any]) -> str:
    if s["success_bounds"]:
        lo, hi = s["success_bounds"]
        undecided = s["verdicts"]["unknown"] + s["verdicts"]["error"]
        succ = f"{lo:.0%}–{hi:.0%} ({undecided} unknown or error)"
    elif s["success_rate"] is None:
        succ = "n/a"
    else:
        lo, hi = s["success_ci95"]
        k, n = s["verdicts"]["success"], s["verdicts"]["success"] + s["verdicts"]["failure"]
        per = s["success_by_sample"]["rates"]
        each = "" if len(per) < 2 else " [" + ", ".join(f"{v:.0%}" for v in per.values()) + "]"
        succ = f"{k}/{n} = {s['success_rate']:.0%} ({lo:.0%}–{hi:.0%}){each}"
    total = s["total_tokens_mean"] if s["user_tokens_recorded"] else f"{s['agent_tokens_mean']}*"
    sel = f"  selection {s['tool_selection']}" if "tool_selection" in s else ""
    return ROW.format(name=name, succ=succ, sel=sel, total=total, **s)


def split_of(recs: list[dict]) -> str:
    """The records' split, 'mixed', or 'unknown' when a task is in no split (an old or renamed task)."""
    found = set()
    for r in recs:
        if "domain" not in r:
            continue
        try:
            found.add(splits.split_of(r["domain"], str(r["task_id"])))
        except KeyError:
            found.add("unknown")
    if len(found) == 1:
        return found.pop()
    return "unknown" if "unknown" in found else "mixed"


def key(rec: dict) -> tuple[str, Any]:
    """A conversation's identity, as `evaluate_record` writes it in a row: (domain:task, sample)."""
    return f"{rec.get('domain')}:{rec.get('task_id')}", rec.get("sample", 0)


def overall(per_file: list[tuple[list[dict], list[dict], str]]) -> tuple[dict | None, str]:
    """One summary over all files, or None and why not.

    Only for one arm on one split (its domains, or a run split over files): pooling arms or
    splits would resample different arms' conversations of a task as one task. Conversations in
    several files count once, the last file's (`latest_records` over all of them)."""
    arms = {r.get("arm") for recs, _, _ in per_file for r in recs}
    split_set = {split for _, _, split in per_file}
    if len(arms) > 1:
        return None, f"files hold different arms ({', '.join(sorted(map(str, arms)))})"
    if len(split_set) > 1 or split_set & {"mixed", "unknown"}:
        return None, f"files are on different or unknown splits ({', '.join(sorted(split_set))})"
    rows_by_key: dict[tuple[str, Any], dict] = {}
    for _, rows, _ in per_file:
        rows_by_key.update({(r["task_id"], r["sample"]): r for r in rows})
    latest = latest_records(r for recs, _, _ in per_file for r in recs)
    broken = [r for r in latest if is_broken(r)]
    rows = [rows_by_key[key(r)] for r in latest if not is_broken(r)]
    return summarise(rows, len(broken), sum(record_cost(r) or 0 for r in broken)), ""


def notes(report: dict[str, Any], crashed: int) -> None:
    """The warnings under the table: crashes left out, oracle errors, agent-only token counts."""
    if "overall" in report:  # a crash re-run in a later file isn't still a crash
        crashed = report["overall"]["crashed_not_scored"]
    if crashed:
        print(f"⚠ {crashed} crashed conversations left out: re-run them for a complete result")
    errors = sum(v["summary"]["verdicts"]["error"] for v in report["files"].values())
    if errors:
        print(f"⚠ the oracle couldn't score {errors} conversations (verdict error): see --json for why")
    if not all(v["summary"]["user_tokens_recorded"] for v in report["files"].values()):
        print(
            "* agent tokens only, here and per million tokens: some records predate the customer's "
            "token count (its cost is included)"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", type=Path, nargs="+", help="rollout files (results/episodes/arms/*.jsonl)")
    p.add_argument("--success", choices=["oracle", "saved"], default="oracle")
    p.add_argument("--judges", choices=["auto", "on", "off"], default="auto", help="the oracle's judges")
    p.add_argument("--tool-selection", metavar="MODEL", help="also run AuditKIT's tool_selection judge")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--json", type=Path, help="write every summary and conversation here")
    a = p.parse_args()

    try:
        from loguru import logger

        logger.remove()  # τ²-bench logs every step
    except ImportError:  # no τ²-bench here: nothing to silence
        pass
    if a.success == "oracle":
        from reflexive.oracle import Tau2Oracle

        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        if a.judges == "on" and not has_key:
            print("⛔ --judges on needs OPENAI_API_KEY")
            return 1
        oracle = Tau2Oracle(judges={"on": True, "off": False}.get(a.judges, has_key))
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
    loaded = []
    for f in a.files:  # read every file before scoring any: a bad file shouldn't waste a paid run
        try:
            loaded.append((f, read_jsonl(f, partial_last_ok=True)))
        except (BadJsonLine, FileNotFoundError) as e:
            print(f"⛔ {e}")
            return 1
    if a.success == "oracle":  # load each domain's τ²-bench scorer once, before the threads
        warm({r["domain"] for _, recs in loaded for r in recs if r.get("domain")}, oracle=True)
    per_file = []
    crashed = 0
    for f, recs in loaded:
        split = split_of(recs)
        rows, s = evaluate(recs, success, judge, a.workers)
        s["split"] = split
        print(line(f.stem, s))
        report["files"][shown(f)] = {"summary": s, "conversations": rows}
        per_file.append((recs, rows, split))
        crashed += s["crashed_not_scored"]
    if len(a.files) > 1:
        both, why = overall(per_file)
        if both is None:
            print(f"overall: not given, {why}")
        else:
            print(line("overall", both))
            report["overall"] = both
    notes(report, crashed)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(report, indent=1) + "\n")
        print(f"→ {shown(a.json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
