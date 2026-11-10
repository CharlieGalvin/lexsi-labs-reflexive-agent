"""Single-turn GRPO from SFT v2 (SEL-21): one sampled agent turn, scored against the teacher's.

On the GPU pod, after ``scripts/pod/setup.sh`` (and ``just grpo-data``):

    just grpo-train --dry-run                        # tokenise and report (no GPU): check first
    just grpo-train --max-steps 2 --report-to none   # smoke run: loads, samples, steps, saves
    just grpo-train                                  # configs/train/grpo_aya32b_v2.yaml
    just grpo-train --push                           # also upload the adapter, private
    just grpo-train other.yaml                       # another config

Loads the config's ``model`` in 4-bit with SFT v2's LoRA ``adapter`` on it (trainable: GRPO
continues it), and trains with TRL's ``GRPOTrainer``: for each prompt it samples
``num_generations`` replies, scores each with `reflexive.grpo_rewards` and moves the adapter
towards the replies that score above their group's mean. Plain TRL, not AgentTune: a sample is
one reply, with no tool run and no customer to answer it.

Writes ``runs/<name>/``: the adapter, ``rewards.jsonl`` (every scored sample, with each reward
component: ``pd.read_json(path, lines=True)``), ``log.jsonl`` (TRL's metrics per step) and
``run_summary.json``. Refuses to overwrite a finished run without ``--force``. Trains on train
tasks only, and checks again.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from reflexive import grpo_data, grpo_rewards, splits
from reflexive.files import ROOT, read_jsonl, sha256, shown
from reflexive.sft import relax_chat_template

HUB_ORG = "reagent-lexsi"
CONFIG = ROOT / "configs" / "train" / "grpo_aya32b_v2.yaml"
REQUIRED = ("name", "model", "adapter", "data", "trainer")
COLUMNS = ("prompt", "target", "target_calls", "domain", "id")  # what the reward reads


def load(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for key in REQUIRED:
        if key not in cfg:
            raise SystemExit(f"⛔ {shown(path)} has no {key!r}")
    if bad := sorted(set(cfg.get("reward_weights") or {}) - set(grpo_rewards.COMPONENTS)):
        raise SystemExit(f"⛔ unknown reward component(s) {bad}: expected some of {grpo_rewards.COMPONENTS}")
    return cfg


def read_rows(data: str) -> list[dict[str, Any]]:
    """The GRPO prompts in ``data`` (a path under the repo), checked: our format, train tasks only."""
    path = ROOT / data
    if not path.exists():
        raise SystemExit(f"⛔ {data} doesn't exist: run `just grpo-data` first")
    rows = read_jsonl(path)
    for r in rows:
        if r.get("format") != grpo_data.FORMAT:
            raise SystemExit(f"⛔ {r.get('id')} isn't {grpo_data.FORMAT}: rebuild with `just grpo-data`")
        splits.assert_not_test(r["domain"], r["task_id"])  # a test task is never trained on
        if splits.split_of(r["domain"], r["task_id"]) != "train":
            raise SystemExit(f"⛔ {r['id']} isn't a train task: rebuild with `just grpo-data`")
    if not rows:
        raise SystemExit(f"⛔ {data} is empty")
    return rows


def prompts(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The training rows, as the dataset's columns. ``target_calls`` as JSON text: Arrow can't
    hold argument dicts of different shapes in one column."""
    rows = read_rows(cfg["data"])
    return [{**{k: r[k] for k in COLUMNS}, "target_calls": json.dumps(r["target_calls"])} for r in rows]


def grpo_args(cfg: dict[str, Any], out: Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """GRPOConfig's keywords: the config's ``trainer`` section (and any overrides) over the fixed ones.

    Checks the batch arithmetic here, so a bad config stops before the 32B model loads: the
    samples generated together (``per_device_train_batch_size`` x ``steps_per_generation``,
    which defaults to ``gradient_accumulation_steps``) and the samples in one training step must
    both hold whole groups."""
    t = {**cfg["trainer"], **(overrides or {})}
    batch = t.get("per_device_train_batch_size", 8)
    accumulate = t.get("gradient_accumulation_steps", 1)
    groups = t.get("num_generations", 8)
    for what, n in (
        (
            "generation batch (per_device_train_batch_size x steps_per_generation)",
            batch * (t.get("steps_per_generation") or accumulate),
        ),
        ("training step (per_device_train_batch_size x gradient_accumulation_steps)", batch * accumulate),
    ):
        if n % groups:
            raise SystemExit(
                f"⛔ {n} samples per {what} don't divide into groups of {groups}: "
                "make it a multiple of num_generations"
            )
    return {
        "output_dir": str(out / "checkpoints"),
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,  # the reward reads target, target_calls, domain, id
        "report_to": ["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        "run_name": cfg["name"],
        **t,
    }


def prompt_tokens(rows: list[dict[str, Any]], tok: Any) -> list[int]:
    """Each prompt's length in tokens, rendered as the trainer renders it (with the turn header)."""
    return [
        len(tok.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True)) for r in rows
    ]


def load_tokenizer(model: str) -> Any:
    """Aya's tokenizer, padding on the left, with the chat template that lets the agent speak first."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, padding_side="left", truncation_side="left")
    tok.chat_template = relax_chat_template(tok.chat_template)  # see reflexive.sft
    return tok


def load_base(cfg: dict[str, Any]) -> Any:
    """The config's model in 4-bit on GPU 0, as SFT trained the adapter on it."""
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = cfg.get("quantization", {})
    return AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=q.get("load_in_4bit", True),
            bnb_4bit_quant_type=q.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=q.get("double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )


def main() -> int:  # noqa: PLR0915 - one linear script: check, load, train, save
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=CONFIG, help=f"default: {shown(CONFIG)}")
    p.add_argument("--dry-run", action="store_true", help="tokenise and report only (no GPU)")
    p.add_argument("--max-steps", type=int, help="override trainer.max_steps (a smoke run: 2)")
    p.add_argument("--report-to", help="override where metrics go (none, wandb)")
    p.add_argument("--force", action="store_true", help="replace a finished run")
    p.add_argument("--push", action="store_true", help=f"upload the adapter to {HUB_ORG}/<name>, private")
    a = p.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation
    os.environ.setdefault("WANDB_PROJECT", "reagent-trainer")  # all our runs in one W&B project

    cfg = load(a.config)
    out = ROOT / "runs" / cfg["name"]
    if (out / "run_summary.json").exists() and not a.force:
        print(f"⛔ {shown(out)} already has a finished run: pass --force to train again")
        return 1
    overrides: dict[str, Any] = {}
    if a.max_steps:
        overrides["max_steps"] = a.max_steps
    if a.report_to:
        overrides["report_to"] = [] if a.report_to == "none" else [a.report_to]
    args = grpo_args(cfg, out, overrides)
    rows = prompts(cfg)

    tok = load_tokenizer(cfg["model"])
    lengths = prompt_tokens(rows, tok)
    n_prompts = args["max_steps"] * args["per_device_train_batch_size"] * args["gradient_accumulation_steps"]
    n_prompts //= args.get("num_generations", 8)
    print(
        f"{len(rows)} prompts, {statistics.median(lengths):,.0f} tokens median, longest {max(lengths):,}; "
        f"{args['max_steps']} steps see {n_prompts} of them, {args.get('num_generations', 8)} samples each"
    )
    if a.dry_run:
        return 0

    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    base = load_base(cfg)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, cfg["adapter"], is_trainable=True)
    model.print_trainable_parameters()

    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w")

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw) -> None:  # noqa: ANN001, ANN003
            if logs:
                log.write(json.dumps({"step": state.global_step, **logs}) + "\n")
                log.flush()

    reward = grpo_rewards.make_reward_fn(cfg.get("reward_weights"), log_path=out / "rewards.jsonl")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward],
        args=GRPOConfig(**args),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
        callbacks=[Log()],
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    result = trainer.train()
    minutes = (time.time() - start) / 60
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.close()

    summary = {
        "name": cfg["name"],
        "model": cfg["model"],
        "adapter": cfg["adapter"],
        "config": shown(a.config),
        "data": {"file": cfg["data"], "sha256": sha256(ROOT / cfg["data"]), "prompts": len(rows)},
        "prompt_tokens": {"median": statistics.median(lengths), "max": max(lengths)},
        "reward_weights": {**grpo_rewards.DEFAULT_WEIGHTS, **(cfg.get("reward_weights") or {})},
        "trainer": {k: v for k, v in args.items() if isinstance(v, str | int | float | bool)},
        "train_loss": result.training_loss,
        "steps": result.global_step,
        "minutes": round(minutes, 1),
        "minutes_per_step": round(minutes / max(result.global_step, 1), 2),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    if a.push:
        repo = f"{HUB_ORG}/{cfg['name']}"
        model.push_to_hub(repo, private=True)
        summary["hub"] = repo
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"done in {minutes:.0f} min ({summary['minutes_per_step']} min/step); "
        f"peak {summary['peak_gpu_gb']} GB; "
        f"adapter in {shown(out)}; per-sample rewards in {shown(out / 'rewards.jsonl')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
