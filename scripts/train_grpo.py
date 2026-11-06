"""GRPO training with AgentTune from a YAML config in configs/train/ (SEL-34). Needs the GPU box.

    python scripts/train_grpo.py configs/train/smoke_add.yaml                  # smoke run
    python scripts/train_grpo.py configs/train/messy_grpo.yaml                 # experiment A
    python scripts/train_grpo.py configs/train/messy_grpo.yaml --max-steps 3 --report-to none
    python scripts/train_grpo.py configs/train/messy_grpo.yaml \\
        --push reagent-lexsi/tiny-aya-messy-grpo-v0

A config has ``task`` (``add_smoke`` or ``messy_lookup``), ``model``,
``output_dir`` and a ``trainer:`` block passed unchanged to ``create_agentic_trainer(algorithm="grpo", ...)``
(AgentTune routes each key to TRL's GRPOConfig or GRPOTrainer). ``base: other.yaml`` starts
from another config and overrides it key by key.

Writes to ``output_dir``: the LoRA adapter, ``trajectories.jsonl`` (AgentTune: one line per
rollout), ``rewards.jsonl`` (ours: one line per scored rollout, with each reward component),
``reward_inputs_debug.json`` (what the reward function received) and ``run_summary.json``
(wall time, seconds per step, peak GPU memory).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import yaml

from reflexive import messy_lookup


def load_config(path: pathlib.Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = load_config(path.parent / cfg.pop("base"))
        trainer = {**base.get("trainer", {}), **cfg.pop("trainer", {})}
        cfg = {**base, **cfg, "trainer": trainer}
    return cfg


def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: first integer
        b: second integer
    """
    return a + b


def says_five(completions, **kwargs) -> list[float]:
    """Smoke-run reward (AgentTune README): 1 if the rollout mentions 5."""
    return [1.0 if "5" in str(c) else 0.0 for c in completions]


def stop_token_ids(model_id: str) -> list[int] | None:
    """Every token that ends the model's turn: its generation_config's eos ids + the tokenizer's.

    AgentTune's training rollouts generate with TRL's generation config, which only stops at
    the tokenizer's eos token. A chat model that ends its turn with another token (Tiny Aya)
    then runs on to max_new_tokens on every turn: junk text, and ~300 s/step instead of tens
    (trial run, 6 Nov). The eval stops correctly because it uses the model's own config.
    """
    from transformers import AutoTokenizer, GenerationConfig

    ids: list[int] = []
    try:
        eos = GenerationConfig.from_pretrained(model_id).eos_token_id
        ids += eos if isinstance(eos, list) else [eos] if eos is not None else []
    except OSError:
        pass
    tok_eos = AutoTokenizer.from_pretrained(model_id).eos_token_id
    if tok_eos is not None:
        ids.append(tok_eos)
    return sorted(set(ids)) or None


def _wandb_url() -> str | None:
    try:
        import wandb
    except ImportError:
        return None
    return wandb.run.url if wandb.run is not None else None


def build_task(cfg: dict, out: pathlib.Path):
    """(train dataset, tools, reward functions) for the config's task."""
    from datasets import Dataset

    task = cfg["task"]
    if task == "add_smoke":
        rows = [{"prompt": [{"role": "user", "content": "What is 2+3?"}]}] * 8
        return Dataset.from_list(rows), [add], [says_five]
    if task == "messy_lookup":
        mod = messy_lookup
        rows = mod.questions("train", styles=cfg.get("task_options", {}).get("styles"))
        ds = Dataset.from_list([
            {"prompt": mod.prompt(r), "answer": r["answer"], "field": r["field"],
             "customer_id": r["customer_id"], "qid": r["qid"], "level": r["level"]}
            for r in rows
        ])
        reward = mod.make_reward_fn(cfg.get("reward_weights"), log_path=out / "rewards.jsonl")
        return ds, mod.TOOLS, [reward]
    raise ValueError(f"unknown task {task!r}: expected add_smoke or messy_lookup")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=pathlib.Path)
    ap.add_argument("--max-steps", type=int, help="override trainer.max_steps")
    ap.add_argument("--output-dir", type=pathlib.Path, help="override output_dir")
    ap.add_argument("--report-to", help="override trainer.report_to (none, wandb, trackio)")
    ap.add_argument("--push", metavar="REPO", help="upload the adapter + logs to this PRIVATE Hub repo")
    a = ap.parse_args()

    cfg = load_config(a.config)
    out = a.output_dir or pathlib.Path(cfg["output_dir"])
    if (out / "run_summary.json").exists():
        ap.error(f"{out} already has a finished run: pass --output-dir for a new one")
    out.mkdir(parents=True, exist_ok=True)

    trainer_kw = dict(cfg.get("trainer", {}))
    if a.max_steps:
        trainer_kw["max_steps"] = a.max_steps
    if a.report_to:
        trainer_kw["report_to"] = a.report_to

    if trainer_kw.get("report_to") == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "reagent-trainer")  # all our runs in one W&B project
        trainer_kw.setdefault("run_name", out.name)

    import torch
    from agenttune.core.backend_factory import create_agentic_trainer

    ds, tools, rewards = build_task(cfg, out)
    if cfg["task"] == "messy_lookup":
        for key, value in messy_lookup.HARNESS.items():  # the same tool wording as eval_lookup.py
            trainer_kw.setdefault(key, value)
    stops = stop_token_ids(cfg["model"])
    gen_kw = trainer_kw.setdefault("generation_kwargs", {})
    if stops and "eos_token_id" not in gen_kw:
        gen_kw["eos_token_id"] = stops
    print(f"stop tokens for generation: {gen_kw.get('eos_token_id')}")
    print(f"task={cfg['task']} model={cfg['model']} train_rows={len(ds)} output={out}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE (CPU!)'}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # peak_gpu_gb excludes vLLM's own memory
    t0 = time.time()
    trainer = create_agentic_trainer(
        algorithm="grpo", backend="trl", model=cfg["model"], train_dataset=ds,
        tools=tools, reward_funcs=rewards, output_dir=str(out), **trainer_kw,
    )
    stats = trainer.train()
    wall = time.time() - t0

    steps = stats.get("total_steps") or trainer_kw.get("max_steps")
    gpu = torch.cuda.is_available()
    summary = {
        "config": str(a.config),
        "task": cfg["task"],
        "model": cfg["model"],
        "steps": steps,
        "wall_seconds": round(wall),
        "seconds_per_step": round(wall / steps, 1) if steps else None,
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1) if gpu else None,
        "final_loss": stats.get("final_loss"),
        "wandb_url": _wandb_url(),
        "hub_repo": a.push,
        "trainer": {k: v for k, v in trainer_kw.items() if isinstance(v, str | int | float | bool | dict)},
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    if a.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.push, private=True, exist_ok=True)
        api.upload_folder(repo_id=a.push, folder_path=str(out), ignore_patterns=["checkpoint-*"])
        print(f"pushed to https://huggingface.co/{a.push} (private)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
