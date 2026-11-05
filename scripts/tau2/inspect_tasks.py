"""Inspect τ²-bench without any API calls: splits, prompt sizes and context fit (SEL-31).

    pip install -e ".[bench]"
    python scripts/tau2/inspect_tasks.py --context 8192

Token counts are approximate (characters / 4); Tiny Aya's own tokenizer will differ by
roughly ±15%. "Gold tool I/O" replays each task's reference actions against a fresh
environment and measures what they return, so it is a *lower bound* on a real rollout.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from typing import Any

from reflexive.episode import official_split

DOMAINS = ["airline", "retail", "telecom"]
CHAT_OVERHEAD = 600  # rough tokens for user / assistant text turns on top of tool I/O


def approx_tokens(obj: Any) -> int:
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str, separators=(",", ":"))
    return len(text) // 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=8192, help="student context window (Tiny Aya: 8192)")
    args = ap.parse_args()

    from tau2.registry import registry

    print(
        f"{'domain':8s} {'train':>5s} {'test':>4s} {'policy':>7s} {'tools':>6s} {'fixed':>6s} "
        f"{'gold I/O med':>12s} {'p90':>5s} {'over ctx':>8s}"
    )
    for d in DOMAINS:
        env = registry.get_env_constructor(d)()
        splits = {s: official_split(d, s) for s in ("train", "test")}
        policy_tok = approx_tokens(env.get_policy())
        tools_tok = approx_tokens([t.openai_schema for t in env.get_tools()])
        fixed = policy_tok + tools_tok

        io = []
        for task in registry.get_tasks_loader(d)("base"):  # "base": the loader's own default
            fresh = registry.get_env_constructor(d)()
            acts = (task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
            total = 0
            for a in acts:
                try:
                    out = fresh.use_tool(a.name, **a.arguments)
                    total += approx_tokens(out) + approx_tokens(a.arguments) + 15
                except Exception:
                    total += 30
            io.append(total)
        over = sum(1 for x in io if fixed + x + CHAT_OVERHEAD > args.context)
        p90 = sorted(io)[int(0.9 * len(io))]
        tr, te = len(splits["train"]), len(splits["test"])
        print(
            f"{d:8s} {tr:>5d} {te:>4d} {policy_tok:7d} {tools_tok:6d} {fixed:6d} "
            f"{int(st.median(io)):12d} {p90:5d} {over:4d}/{len(io)}"
        )


if __name__ == "__main__":
    main()
