"""Check the single-turn GRPO reward on the real prompts (SEL-21). No API calls, no GPU.

    just grpo-reward-check                                   # data/train/grpo-healable.jsonl
    just grpo-reward-check data/train/grpo-v2.jsonl --weights '{"format": 0.5}'

Two checks, before spending GPU time on the reward:

1. **The teacher's own turn scores the maximum** on every prompt. Anything less is a bug in the
   reward (or a reference that shouldn't be trained on), listed with what was found. Exit 1.
2. **Typical mistakes score less**, shown as a table: each prompt's teacher turn with one thing
   broken (an invented ID, the wrong tool, a missing argument, the call inside prose, a
   hand-off, "OK.", a question to the customer instead, a customer-only tool on telecom).
   GRPO learns from differences within a group, so each row must sit clearly below the maximum.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reflexive import grpo_rewards
from reflexive.files import ROOT, read_jsonl, shown

DATA = ROOT / "data" / "train" / "grpo-healable.jsonl"
Row = dict[str, Any]
CUSTOMER_TOOL = {"telecom": "toggle_airplane_mode"}  # a tool only the customer's phone has


def action(calls: list[Row]) -> str:
    body = json.dumps([{"tool_name": c["name"], "parameters": c["arguments"]} for c in calls], indent=4)
    return f"Action: ```json\n{body}\n```"


def _invented_id(r: Row, calls: list[Row]) -> str | None:
    keys = [k for k in calls[0]["arguments"] if k.endswith("id")]
    if not keys:
        return None
    calls[0]["arguments"][keys[0]] = "X9Z8Q7"
    return action(calls)


def _wrong_tool(r: Row, calls: list[Row]) -> str | None:
    used = {c["name"] for c in r["target_calls"]}
    other = next((n for n in grpo_rewards.agent_tool_names(r["domain"]) if n not in used), None)
    if other is None:
        return None
    calls[0]["name"] = other
    return action(calls)


def _missing_arg(r: Row, calls: list[Row]) -> str | None:
    if not calls[0]["arguments"]:
        return None
    calls[0]["arguments"].pop(next(iter(calls[0]["arguments"])))
    return action(calls)


MISTAKES: dict[str, Callable[[Row, list[Row]], str | None]] = {
    "invented ID": _invented_id,
    "wrong tool": _wrong_tool,
    "missing argument": _missing_arg,
    "call inside prose": lambda r, c: f"Sure, let me check that for you. {action(c)}\nTool result: {{}}",
    "hand-off": lambda r, c: action([{"name": "transfer_to_human_agents", "arguments": {"summary": "help"}}]),
    '"OK."': lambda r, c: "OK.",
    "asks instead": lambda r, c: "Could you please confirm the details you'd like me to use for this?",
    "customer-only tool": lambda r, c: (
        action([{"name": CUSTOMER_TOOL[r["domain"]], "arguments": {}}])
        if r["domain"] in CUSTOMER_TOOL
        else None
    ),
}


def score(r: Row, text: str) -> Row:
    return grpo_rewards.score(text, r["target_calls"], r["domain"], r["prompt"], r["target"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", type=Path, nargs="?", default=DATA, help="grpo-*.jsonl from `just grpo-data`")
    p.add_argument("--weights", type=json.loads, default={}, help="JSON: weights to override")
    a = p.parse_args()

    if not a.data.exists():
        print(f"⛔ {shown(a.data)} doesn't exist: run `just grpo-data` first")
        return 1
    rows = [r for r in read_jsonl(a.data) if r["target_calls"]]
    weights = {**grpo_rewards.DEFAULT_WEIGHTS, **a.weights}
    best = sum(weights.values())

    below = []
    for r in rows:
        s = score(r, r["target"])
        if (t := grpo_rewards.total(s, weights)) < best - 1e-9:
            below.append((r["id"], t, [k for k in grpo_rewards.COMPONENTS if s[k] < 1], s["findings"]))
    print(f"1. the teacher's turn: {len(rows) - len(below)}/{len(rows)} prompts score the maximum, {best:g}")
    for rid, t, comps, found in below[:20]:
        print(f"   ⛔ {rid}: {t:.2f} ({', '.join(comps)}; found {found})")

    print(f"\n2. typical mistakes (mean over the prompts where each applies; maximum {best:g})\n")
    head = " | ".join(f"{k:>10}" for k in grpo_rewards.COMPONENTS)
    print(f"   {'mistake':<20} {'prompts':>7} {'reward':>7} | {head}")
    for name, make in MISTAKES.items():
        scored = [score(r, t) for r in rows if (t := make(r, copy.deepcopy(r["target_calls"]))) is not None]
        if not scored:
            continue
        mean = sum(grpo_rewards.total(s, weights) for s in scored) / len(scored)
        comps = " | ".join(
            f"{sum(s[k] for s in scored) / len(scored):>10.2f}" for k in grpo_rewards.COMPONENTS
        )
        print(f"   {name:<20} {len(scored):>7} {mean:>7.2f} | {comps}")
    return 1 if below else 0


if __name__ == "__main__":
    raise SystemExit(main())
