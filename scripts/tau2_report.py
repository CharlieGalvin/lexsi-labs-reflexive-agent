"""Print reward, token usage and cost per task for a τ²-bench run (used by tau2_first_run.sh).

    python scripts/tau2_report.py <save-to name or path>

Cost is the sum of each message's own cost. τ²-bench's per-run `agent_cost` is left empty
whenever any one agent message has no cost (in v2.0.0 the scripted greeting never has one),
so it isn't used. If no message has a cost, it is estimated from tokens at GPT-4.1 list
prices. Your provider dashboard shows actual spend.
"""

from __future__ import annotations

import json
import pathlib
import sys

IN_PRICE, OUT_PRICE = 2e-06, 8e-06  # USD per token: GPT-4.1, as listed in LiteLLM's price map


def search_roots(name: str) -> list[pathlib.Path]:
    roots = [pathlib.Path(name)]
    try:
        from tau2.utils.utils import DATA_DIR  # where `tau2 run --save-to` writes

        roots.append(pathlib.Path(DATA_DIR) / "simulations" / name)
    except ImportError:
        roots.append(pathlib.Path("data/simulations") / name)
    return roots


def find_files(name: str) -> list[pathlib.Path]:
    for root in search_roots(name):
        if root.is_file():
            return [root]
        if root.is_dir():
            return sorted(root.rglob("*.json"))
    return []


def fmt_cost(summed: float | None, tokens: list[int]) -> str:
    if summed is not None:
        return f"${summed:.4f}"
    return f"~${tokens[0] * IN_PRICE + tokens[1] * OUT_PRICE:.4f} (estimated)"


def main() -> None:
    files = find_files(sys.argv[1] if len(sys.argv) > 1 else "")
    if not files:
        looked = ", ".join(str(r) for r in search_roots(sys.argv[1] if len(sys.argv) > 1 else ""))
        sys.exit(f"No results found for {sys.argv[1:]!r} (looked in {looked})")
    total, n = 0.0, 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        sims = data.get("simulations", []) if isinstance(data, dict) else []
        for sim in sims:
            tok = {"assistant": [0, 0], "user": [0, 0]}
            cost: dict[str, float | None] = {"assistant": None, "user": None}
            for m in sim.get("messages", []):
                usage, role = m.get("usage") or {}, m.get("role")
                if role in tok:
                    tok[role][0] += usage.get("prompt_tokens") or 0
                    tok[role][1] += usage.get("completion_tokens") or 0
                    if m.get("cost") is not None:
                        cost[role] = (cost[role] or 0.0) + m["cost"]
            reward = (sim.get("reward_info") or {}).get("reward")
            print(f"task {sim.get('task_id')}: reward={reward}")
            for label, role in (("agent:   ", "assistant"), ("customer:", "user")):
                t = tok[role]
                print(f"   {label} {t[0]:>7,} in / {t[1]:>6,} out tokens   cost {fmt_cost(cost[role], t)}")
            total += sum(c for c in cost.values() if c is not None)
            n += 1
    if n:
        print(f"total: ${total:.4f} for {n} task(s), ${total / n:.4f} per task")


if __name__ == "__main__":
    main()
