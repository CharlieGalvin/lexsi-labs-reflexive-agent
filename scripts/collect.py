"""Roll out a model on the held-out eval split and save trajectories.

    # baseline (before)
    python scripts/collect.py --config configs/grpo_efficiency.yaml --out results/base.jsonl
    # tuned (after)
    python scripts/collect.py --config configs/grpo_efficiency.yaml \
        --adapter runs/grpo-eff-v0 --out results/grpo-eff-v0.jsonl

Both runs use the same harness, prompt, eval split and decoding settings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from reflexive import load_config
from reflexive.data import load_task
from reflexive.rollout import hf_generate_fn, run_episode
from reflexive.tools import get_tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir or Hub id (omit for base)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    ev = cfg["eval"]

    tools = get_tools(cfg["tools"])
    ds = load_task(cfg["data"], "eval")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    gen = hf_generate_fn(cfg["model"]["base"], args.adapter,
                         max_new_tokens=ev["max_new_tokens"], temperature=ev["temperature"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in tqdm(ds, desc="rollouts"):
            ep = run_episode(gen, tools, row["prompt"], max_steps=ev["max_steps"])
            fh.write(json.dumps({"id": row["id"], "task": row["prompt"][-1]["content"],
                                 "answer": row["answer"], "model": cfg["model"]["base"],
                                 "adapter": args.adapter, **ep}, default=str) + "\n")
    print(f"[collect] wrote {len(ds)} trajectories -> {out}")


if __name__ == "__main__":
    main()
