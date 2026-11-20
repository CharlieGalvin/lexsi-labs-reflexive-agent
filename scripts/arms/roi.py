"""Latency, cost and success per arm and domain, from the saved conversations (SEL-86, SEL-108).

    python scripts/arms/roi.py A S2 GPT --gpu-hourly 2.99          # on the pod, test split
    python scripts/arms/roi.py A C S2 --split test --json results/roi-test.metrics.json

For each arm (results/episodes/arms/<arm>-<domain>-<split>.jsonl), scored conversations only:

* **Seconds per agent request**: a conversation's model time ÷ its model calls. Reported as the
  median and the mean over conversations (the README's medians are this median), and the pooled
  mean (all model time ÷ all calls).
* **Cost per conversation** = the agent + the simulated customer (``user_cost``, priced by τ²-bench).
  The agent is priced by how it runs:
    - an API model at its list price per token (``API_PRICES``: Aya via Cohere, GPT-4.1);
    - a self-hosted model (S2, P, …) at ``--gpu-hourly`` × its model time ÷ ``--concurrency``,
      since one GPU serves that many conversations at once. That's an estimate: say so.
* **Cost per solved task** = total cost ÷ conversations passed.

Success here is the records' own verdict. If a run was rolled out with judges off (arm GPT), its
retail and airline successes are provisional until re-scored with judges on (`just arm-eval`).
No GPU and no API calls.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from reflexive.metrics import is_broken

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
DOMAINS = ("retail", "airline", "telecom")
# USD per input and output token, for agents reached through an API
API_PRICES = {
    "A": (0.50e-6, 1.50e-6),  # Aya Expanse 32B via Cohere (reflexive.teacher PRICE_IN / PRICE_OUT)
    "C": (0.50e-6, 1.50e-6),
    "GPT": (2.00e-6, 8.00e-6),  # GPT-4.1 list price, passed through by OpenRouter
}


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        r for r in (json.loads(x) for x in path.read_text().splitlines() if x.strip()) if not is_broken(r)
    ]


def agent_cost(rec: dict, arm: str, gpu_hourly: float | None, concurrency: int) -> float | None:
    u = rec.get("agent_usage") or {}
    if arm in API_PRICES:
        p_in, p_out = API_PRICES[arm]
        return u.get("prompt_tokens", 0) * p_in + u.get("completion_tokens", 0) * p_out
    if gpu_hourly is None:
        return None
    return gpu_hourly * u.get("seconds", 0.0) / 3600 / concurrency


def summarise(recs: list[dict], arm: str, gpu_hourly: float | None, concurrency: int) -> dict | None:
    if not recs:
        return None
    per_request = [
        (r["agent_usage"]["seconds"] / r["agent_usage"]["calls"])
        for r in recs
        if (r.get("agent_usage") or {}).get("calls")
    ]
    seconds = sum((r.get("agent_usage") or {}).get("seconds", 0.0) for r in recs)
    calls = sum((r.get("agent_usage") or {}).get("calls", 0) for r in recs)
    passed = sum(bool(r["outcome"]["passed"]) for r in recs)
    costs = [agent_cost(r, arm, gpu_hourly, concurrency) for r in recs]
    agent = None if any(c is None for c in costs) else sum(c or 0.0 for c in costs)
    customer = [float(r.get("user_cost") or 0.0) for r in recs]
    total = None if agent is None else agent + sum(customer)
    judged = [r["outcome"].get("judged") for r in recs]
    skipped = sum(bool(r["outcome"].get("skipped")) for r in recs)
    return {
        "conversations": len(recs),
        "passed": passed,
        "success": passed / len(recs),
        "judges_skipped_in": skipped,
        "judged": sum(bool(j) for j in judged),
        "seconds_per_request_median": statistics.median(per_request) if per_request else None,
        "seconds_per_request_mean": statistics.fmean(per_request) if per_request else None,
        "seconds_per_request_pooled": seconds / calls if calls else None,
        "agent_cost_usd": agent,
        "customer_cost_usd": sum(customer),
        "cost_per_conversation_usd": None if total is None else total / len(recs),
        "cost_per_solved_task_usd": None if total is None or not passed else total / passed,
    }


def fmt(x: float | None, spec: str) -> str:
    return "—" if x is None else format(x, spec)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+", help="arm keys, e.g. A S2 GPT")
    p.add_argument("--split", default="test", choices=["train", "dev", "val", "test"])
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--gpu-hourly", type=float, help="USD per hour for the self-hosted model's GPU (H100)")
    p.add_argument(
        "--concurrency", type=int, default=4, help="conversations sharing the GPU at once (rollout --workers)"
    )
    p.add_argument("--json", type=Path, help="also save the table here")
    a = p.parse_args(argv)

    rows: dict[str, dict] = {}
    widths = (5, 8, 4, 8, 16, 6, 6, 14, 13)
    names = (
        "Arm",
        "domain",
        "n",
        "success",
        "s/request median",
        "mean",
        "pooled",
        "$/conversation",
        "$/solved task",
    )

    def line(cells: tuple[str, ...]) -> str:
        return "  ".join(
            c.ljust(w) if i < 2 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths, strict=True))
        )

    print(line(names))
    for arm in a.arms:
        allrecs = []
        for d in (*DOMAINS, "all"):
            recs = allrecs if d == "all" else load(a.episodes / f"{arm}-{d}-{a.split}.jsonl")
            if d != "all":
                allrecs = allrecs + recs
            s = summarise(recs, arm, a.gpu_hourly, a.concurrency)
            if s is None:
                print(f"{arm:<5} {d:<8} (no records)")
                continue
            rows[f"{arm}/{d}"] = s
            note = (
                "  (judges off: provisional)"
                if s["judges_skipped_in"] and d in ("retail", "airline", "all")
                else ""
            )
            cells = (
                arm,
                d,
                str(s["conversations"]),
                f"{s['success']:.1%}",
                fmt(s["seconds_per_request_median"], ".1f"),
                fmt(s["seconds_per_request_mean"], ".1f"),
                fmt(s["seconds_per_request_pooled"], ".1f"),
                fmt(s["cost_per_conversation_usd"], ".3f"),
                fmt(s["cost_per_solved_task_usd"], ".2f"),
            )
            print(line(cells) + note)
    self_hosted = [x for x in a.arms if x not in API_PRICES]
    if self_hosted and a.gpu_hourly is None:
        print(f"\n⚠ no --gpu-hourly: no cost for the self-hosted arms ({', '.join(self_hosted)})")
    if a.json:
        a.json.write_text(
            json.dumps(
                {
                    "split": a.split,
                    "gpu_hourly_usd": a.gpu_hourly,
                    "concurrency": a.concurrency,
                    "api_prices_usd_per_token": {k: v for k, v in API_PRICES.items() if k in a.arms},
                    "notes": (
                        "cost = agent + customer; self-hosted agent = GPU $/h × model time ÷ concurrency"
                        " (estimate); seconds per request = model time ÷ calls per conversation"
                        " (median, mean) and pooled"
                    ),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nsaved {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
