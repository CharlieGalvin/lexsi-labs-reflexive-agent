"""Score saved trajectories: our headline metrics + AuditKIT's agent scorers.

    python scripts/evaluate.py results/base.jsonl results/grpo-eff-v0.jsonl

Writes <name>.metrics.json (and <name>.auditkit.json with the full AuditKIT run)
next to each input file. No model or GPU needed.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from reflexive.metrics import aggregate, episode_metrics

DEFAULT_SCORERS = ["redundant_tool_calls", "agent_loop_detection", "tool_call_validity",
                   "tool_selection"]


def auditkit_scores(records: list[dict], scorers: list[str], tools: list[str]):
    from auditkit.agent_eval import (
        AgentCase,
        AgentEvalRunner,
        AgentEvalSpec,
        AnswerAssertion,
        episode_from_openai_messages,
    )

    episodes, cases = [], []
    for r in records:
        episodes.append(episode_from_openai_messages(r["messages"], case_id=r["id"],
                                                     final_answer=r.get("final_answer") or ""))
        cases.append(AgentCase(id=r["id"], task=r["task"], category="tool_use",
                               allowed_tools=tools,
                               outcome=AnswerAssertion(reference=r["answer"], mode="contains")))
    result = AgentEvalRunner().run(AgentEvalSpec(cases=cases, mode="recorded",
                                                 episodes=episodes, scorers=scorers, trials=1))
    per_scorer: dict[str, list[float]] = defaultdict(list)
    for row in result.rows:
        for name, d in (row.diagnostics or {}).items():
            v = d.get("value") if isinstance(d, dict) else None
            if isinstance(v, (int, float)):
                per_scorer[name].append(float(v))
    summary = {f"auditkit/{k}": round(mean(v), 4) for k, v in sorted(per_scorer.items())}
    return summary, result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--scorers", nargs="*", default=DEFAULT_SCORERS)
    ap.add_argument("--tools", nargs="*", default=["query_biogrid"])
    ap.add_argument("--no-auditkit", action="store_true")
    args = ap.parse_args()

    for f in args.files:
        path = Path(f)
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        metrics = aggregate([episode_metrics(r["messages"], r["answer"]) for r in records])
        if not args.no_auditkit:
            ak_summary, ak_result = auditkit_scores(records, args.scorers, args.tools)
            metrics.update(ak_summary)
            ak_result.to_json(str(path.with_suffix(".auditkit.json")))
        out = path.with_suffix(".metrics.json")
        out.write_text(json.dumps(metrics, indent=2))
        print(f"\n== {path.name} ==")
        for k, v in metrics.items():
            print(f"  {k:36s} {v}")


if __name__ == "__main__":
    main()
