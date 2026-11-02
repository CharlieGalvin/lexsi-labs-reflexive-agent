"""Train the agent with GRPO (or any AgentTune algorithm) on tool-calling rollouts.

    python scripts/train.py --config configs/grpo_efficiency.yaml
    python scripts/train.py --config configs/grpo_efficiency.yaml train.max_steps=10   # smoke test

Writes to train.output_dir: the LoRA adapter, trajectories.jsonl (every rollout, readable
by AuditKIT) and lexsi_provenance.json. The resolved config is saved alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflexive import load_config
from reflexive.data import load_task
from reflexive.rewards import build_reward_funcs
from reflexive.tools import get_tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted key=value overrides")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)

    from agenttune.core.backend_factory import create_agentic_trainer

    tools = get_tools(cfg["tools"])
    reward_funcs, weights = build_reward_funcs(cfg["rewards"], [t.__name__ for t in tools])
    train_ds = load_task(cfg["data"], "train")

    train_kw = dict(cfg["train"])
    out = Path(train_kw["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[train] {len(train_ds)} rows | rewards={[f.__name__ for f in reward_funcs]} "
          f"weights={weights} | -> {out}")

    trainer = create_agentic_trainer(
        model=cfg["model"]["base"],
        train_dataset=train_ds,
        tools=tools,
        reward_funcs=reward_funcs,
        reward_weights=weights,   # AgentTune folds these into one combined reward
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 42),
        **train_kw,
    )
    results = trainer.train()
    print(f"[train] done: {results}")


if __name__ == "__main__":
    main()
