"""Print reward, token usage and cost per task for a τ²-bench run (used by tau2_first_run.sh).

    python scripts/tau2_report.py <save-to name or path>

τ²-bench prices each message with LiteLLM. Some model IDs (e.g. OpenRouter ones) are missing
from LiteLLM's price list, so the reported cost can be empty; in that case cost is estimated
from token counts at GPT-4.1 list prices. Check your provider dashboard for actual spend.
"""

from __future__ import annotations

import json
import pathlib
import sys

IN_PRICE, OUT_PRICE = 2e-06, 8e-06  # USD per token: GPT-4.1, as listed in LiteLLM's price map


def find_files(name: str) -> list[pathlib.Path]:
    for root in (pathlib.Path(name), pathlib.Path("data/simulations") / name):
        if root.is_file():
            return [root]
        if root.is_dir():
            return sorted(root.rglob("*.json"))
    return []


def fmt_cost(reported: float | None, tokens: list[int]) -> str:
    if reported:
        return f"${reported:.4f}"
    return f"~${tokens[0] * IN_PRICE + tokens[1] * OUT_PRICE:.4f} (estimated)"


def main() -> None:
    files = find_files(sys.argv[1] if len(sys.argv) > 1 else "")
    if not files:
        sys.exit(f"No results found for {sys.argv[1:]!r} (looked in . and data/simulations/)")
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        sims = data.get("simulations", []) if isinstance(data, dict) else []
        for sim in sims:
            tok = {"assistant": [0, 0], "user": [0, 0]}
            for m in sim.get("messages", []):
                usage, role = m.get("usage") or {}, m.get("role")
                if role in tok:
                    tok[role][0] += usage.get("prompt_tokens") or 0
                    tok[role][1] += usage.get("completion_tokens") or 0
            reward = (sim.get("reward_info") or {}).get("reward")
            a, u = tok["assistant"], tok["user"]
            print(f"task {sim.get('task_id')}: reward={reward}")
            for label, t, key in (("agent:   ", a, "agent_cost"), ("customer:", u, "user_cost")):
                print(f"   {label} {t[0]:>7,} in / {t[1]:>6,} out tokens   cost {fmt_cost(sim.get(key), t)}")


if __name__ == "__main__":
    main()
