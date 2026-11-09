"""Replay saved rollouts through the runtime monitor, turn by turn, as if it had run live (SEL-69).

    uv run python scripts/arms/monitor_replay.py results/episodes/arms/A-retail-train.jsonl ...
    uv run python scripts/arms/monitor_replay.py FILES --json results/monitor-replay.json

No new conversations: each record's ``agent_view`` (the conversation as the agent saw it) is fed
to ``reflexive.monitor.Watcher("act", tools)`` one agent turn at a time, with the domain's tools
from ``reflexive.hallucination.domain_tools``. For each conversation it reports where
``no_progress`` fires, what the guard would have done in ``act`` mode (nudge, then stop), and the
conversation's termination and pass, so a false alarm (a fire on a conversation that passed, or
one that didn't loop to the step limit) is easy to see.

Case ids are the arm letter and the record's line number in its file (``A37`` = line 37, from 0),
as in tests/fixtures/no_progress_cases.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from reflexive.hallucination import domain_tools
from reflexive.monitor import Watcher, is_call, normalise


def longest_repeat_run(view: list[dict]) -> int:
    """The longest run of text-only agent turns repeating earlier text (no_progress fires at 3)."""
    said: set[str] = set()
    run = best = 0
    for m in view:
        if m.get("role") != "assistant":
            continue
        text = normalise(m.get("content"))
        run = run + 1 if not is_call(m) and text and text in said else 0
        best = max(best, run)
        if text:
            said.add(text)
    return best


def replay(rec: dict) -> dict:
    """One conversation through a live-style Watcher: the no_progress fires and the guard's reaction."""
    tools, customer = domain_tools(rec["domain"])
    w = Watcher("act", tools, customer_tools=customer)
    view = rec["agent_view"]
    reactions = []
    for i, m in enumerate(view):
        if m.get("role") != "assistant":
            continue
        # the monitor runs once the agent's turn and its tool results (if any) are in
        j = i + 1
        while j < len(view) and view[j].get("role") == "tool":
            j += 1
        r = w.after_turn(view[:j])
        if r:
            reactions.append(r)
    if w.error:
        raise RuntimeError(f"monitor error on {rec['domain']} {rec['task_id']}: {w.error}")
    fires = [a.agent_turn for a in w.monitor.alarms if a.kind == "no_progress"]
    return {
        "agent_turns": sum(m.get("role") == "assistant" for m in view),
        "longest_repeat_run": longest_repeat_run(view),
        "fires_at_agent_turn": fires,
        "nudged_at": w.guard.nudged_at,
        "stopped_at": w.guard.stopped_at,
    }


def verdict(row: dict) -> str:
    """How a conversation's fires line up with what happened."""
    loop = row["termination"] == "max_steps"
    fired = bool(row["fires_at_agent_turn"])
    if fired and row["passed"]:
        return "false alarm: passed"
    if fired and not loop:
        return "fired, not a step-limit loop"
    if loop and not fired:
        return "missed: step-limit, no fire"
    return "caught" if fired else "quiet"


def summary(by_file: dict[str, list[dict]]) -> dict:
    """Per file: counts, verdicts, the conversations that fired, and how close the rest came."""
    out = {"check": "no_progress replay (reflexive.monitor.Watcher, act mode)", "files": {}}
    for f, rs in by_file.items():
        out["files"][f] = {
            "conversations": len(rs),
            "step_limit": sum(r["termination"] == "max_steps" for r in rs),
            "passed": sum(r["passed"] for r in rs),
            "terminations": dict(collections.Counter(r["termination"] for r in rs)),
            "verdicts": dict(collections.Counter(r["verdict"] for r in rs)),
            "fired": [
                {
                    k: r[k]
                    for k in (
                        "id",
                        "task_id",
                        "termination",
                        "passed",
                        "agent_turns",
                        "fires_at_agent_turn",
                        "nudged_at",
                        "stopped_at",
                    )
                }
                for r in rs
                if r["fires_at_agent_turn"]
            ],  # fmt: skip
            "longest_repeat_run": dict(
                sorted(collections.Counter(r["longest_repeat_run"] for r in rs).items())
            ),
            "max_agent_turns": max(r["agent_turns"] for r in rs),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--json", type=Path, help="write every conversation's row here")
    p.add_argument("--metrics", type=Path, help="write the per-file summary here (results/*.metrics.json)")
    a = p.parse_args(argv)

    rows = []
    for path in a.files:
        for n, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            rec = json.loads(line)
            row = {
                "file": path.name,
                "id": f"{rec.get('arm', '?')}{n}",
                "domain": rec["domain"],
                "task_id": rec["task_id"],
                "sample": rec.get("sample", 0),
                "termination": rec.get("termination"),
                "passed": bool((rec.get("outcome") or {}).get("passed")),
                **replay(rec),
            }
            row["verdict"] = verdict(row)
            rows.append(row)

    by_file = collections.defaultdict(list)
    for r in rows:
        by_file[r["file"]].append(r)
    for f, rs in by_file.items():
        c = collections.Counter(r["verdict"] for r in rs)
        loops = sum(r["termination"] == "max_steps" for r in rs)
        passes = sum(r["passed"] for r in rs)
        print(f"\n{f}: {len(rs)} conversations, {loops} to the step limit, {passes} passed")
        print("  " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        for r in rs:
            if r["fires_at_agent_turn"] or r["termination"] == "max_steps":
                act = (
                    f"nudge @{r['nudged_at']}" + (f", stop @{r['stopped_at']}" if r["stopped_at"] else "")
                    if r["nudged_at"]
                    else "-"
                )
                print(
                    f"  {r['id']:>4} task {str(r['task_id'])[:40]:<40} {r['termination']:<10} "
                    f"passed={r['passed']!s:<5} turns={r['agent_turns']:<3} "
                    f"fires={r['fires_at_agent_turn'][:6]}{'…' if len(r['fires_at_agent_turn']) > 6 else ''} "
                    f"act: {act}  [{r['verdict']}]"
                )
    if a.metrics:
        a.metrics.parent.mkdir(parents=True, exist_ok=True)
        a.metrics.write_text(json.dumps(summary(by_file), indent=1) + "\n")
        print(f"wrote {a.metrics}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
