"""Indicative SFT run (SEL-65): QLoRA on Aya Expanse 32B, learning the agent's turns of the fixes.

On the GPU pod, after ``scripts/pod/setup.sh``:

    just sft-train --dry-run      # tokenise and report (no GPU): check the data first
    just sft-train                # configs/train/sft_aya32b.yaml; ~1 hour on one H100
    just sft-train --push         # also upload the adapter to the team org, private
    just sft-train other.yaml     # another config

Reads ``data/train/sft-all.jsonl`` (`just train-data --select all`), trains on the ``healable``
examples and measures loss on the ``solved_by_aya`` ones (held out: not trained on). Labels come
from `reflexive.sft.encode`: only the agent's turns are targets. Writes ``runs/<name>/``: the
adapter, ``run_summary.json`` (data, settings, losses, time, peak memory) and ``log.jsonl``.
Refuses to overwrite a finished run without ``--force``. Never reads dev or test tasks:
`just train-data` only writes train tasks, and this checks again.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from reflexive import splits
from reflexive.files import ROOT, read_jsonl, sha256, shown
from reflexive.sft import encode_all, pad_batch, relax_chat_template

HUB_ORG = "reagent-lexsi"
CONFIG = ROOT / "configs" / "train" / "sft_aya32b.yaml"


def load(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    for key in ("name", "model", "data", "train_group", "eval_group", "max_len", "lora", "trainer"):
        if key not in cfg:
            raise SystemExit(f"⛔ {shown(path)} has no {key!r}")
    return cfg


def examples(cfg: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(ROOT / cfg["data"])
    for r in rows:
        splits.assert_not_test(r["domain"], r["task_id"])  # a test task is never trained on
        if splits.split_of(r["domain"], r["task_id"]) != "train":
            raise SystemExit(f"⛔ {r['id']} isn't a train task: rebuild with `just train-data`")
    train = [r for r in rows if r["group"] == cfg["train_group"]]
    held = [r for r in rows if r["group"] == cfg["eval_group"]]
    if not train:
        raise SystemExit(f"⛔ no {cfg['train_group']} examples in {cfg['data']}")
    return train, held


def main() -> int:  # noqa: C901, PLR0915 - one linear script: load, tokenise, train, save
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=CONFIG, help=f"default: {shown(CONFIG)}")
    p.add_argument("--dry-run", action="store_true", help="tokenise and report only (no GPU)")
    p.add_argument("--force", action="store_true", help="replace a finished run")
    p.add_argument("--push", action="store_true", help=f"upload the adapter to {HUB_ORG}/<name>, private")
    a = p.parse_args()

    cfg = load(a.config)
    out = ROOT / "runs" / cfg["name"]
    if (out / "adapter_config.json").exists() and not a.force:
        print(f"⛔ {shown(out)} already has an adapter: pass --force to train again")
        return 1
    train_rows, held_rows = examples(cfg)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.chat_template = relax_chat_template(tok.chat_template)  # the agent speaks first: see reflexive.sft
    train_enc, train_stats = encode_all(train_rows, tok, cfg["max_len"])
    held_enc, held_stats = encode_all(held_rows, tok, cfg["max_len"])
    print(
        f"train: {train_stats['examples']} conversations, {train_stats['tokens']['total']:,} tokens "
        f"({train_stats['target_share']:.0%} targets), longest {train_stats['tokens']['max']:,}; "
        f"held out: {held_stats['examples']}"
    )
    if train_stats["too_long"]:
        print(f"⚠ left out (over {cfg['max_len']} tokens): {train_stats['too_long']}")
    if a.dry_run:
        return 0

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    q = cfg.get("quantization", {})
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=q.get("load_in_4bit", True),
            bnb_4bit_quant_type=q.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=q.get("double_quant", True),
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lc = cfg["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}

    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w")

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw) -> None:  # noqa: ANN001, ANN003
            if logs:
                log.write(json.dumps({"step": state.global_step, **logs}) + "\n")
                log.flush()

    t = dict(cfg["trainer"])
    eval_steps = t.pop("eval_steps", None)
    wandb = bool(os.environ.get("WANDB_API_KEY"))
    args = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps" if held_enc and eval_steps else "no",
        eval_steps=eval_steps,
        per_device_eval_batch_size=1,
        save_strategy="no",
        remove_unused_columns=False,
        report_to=["wandb"] if wandb else [],
        run_name=cfg["name"],
        **t,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_enc,
        eval_dataset=held_enc or None,
        data_collator=collate,
        callbacks=[Log()],
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    first_eval = trainer.evaluate() if held_enc else {}
    result = trainer.train()
    last_eval = trainer.evaluate() if held_enc else {}
    minutes = (time.time() - start) / 60
    model.save_pretrained(out)
    tok.save_pretrained(out)
    log.close()

    summary = {
        "name": cfg["name"],
        "model": cfg["model"],
        "config": shown(a.config),
        "data": {
            "file": cfg["data"],
            "sha256": sha256(ROOT / cfg["data"]),
            "train": train_stats,
            "held_out": held_stats,
        },
        "lora": lc,
        "trainer": cfg["trainer"],
        "train_loss": result.training_loss,
        "held_out_loss": {"before": first_eval.get("eval_loss"), "after": last_eval.get("eval_loss")},
        "steps": result.global_step,
        "minutes": round(minutes, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0),
    }
    if a.push:
        repo = f"{HUB_ORG}/{cfg['name']}"
        model.push_to_hub(repo, private=True)
        summary["hub"] = repo
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        f"done in {minutes:.0f} min: train loss {result.training_loss:.3f}; held-out loss "
        f"{summary['held_out_loss']['before']} -> {summary['held_out_loss']['after']}; "
        f"peak {summary['peak_gpu_gb']} GB; adapter in {shown(out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
