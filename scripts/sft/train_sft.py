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

A heal cycle (SEL-68) runs it through ``scripts/heal/train_and_serve.sh`` with
``configs/train/sft_heal.yaml`` and ``--data`` (the cycle's fixes), ``--out`` (the folder the cycle
reads the adapter from), ``--name`` and, from the second cycle on, ``--init-adapter`` (the adapter
being healed, trained further rather than replaced).
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any

import yaml

from reflexive import splits
from reflexive.adapters import git_state
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
    groups = cfg["train_group"] if isinstance(cfg["train_group"], list) else [cfg["train_group"]]
    train = [r for r in rows if r["group"] in groups]
    held = [r for r in rows if r["group"] == cfg["eval_group"]]
    if not train:
        raise SystemExit(f"⛔ no {cfg['train_group']} examples in {cfg['data']}")
    return train, held


def trainer_args(cfg: dict[str, Any], out: Path, has_held_out: bool, bf16: bool) -> dict[str, Any]:
    """TrainingArguments' keywords: the config's ``trainer`` section plus the fixed ones.

    transformers 5 dropped ``warmup_ratio``: ``warmup_steps`` below 1 is now the ratio."""
    from transformers import TrainingArguments

    t = dict(cfg["trainer"])
    eval_steps = t.pop("eval_steps", None)
    known = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in t and "warmup_ratio" not in known:
        t["warmup_steps"] = t.pop("warmup_ratio")
    return {
        "output_dir": str(out / "checkpoints"),
        "bf16": bf16,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "steps" if has_held_out and eval_steps else "no",
        "eval_steps": eval_steps,
        "per_device_eval_batch_size": 1,
        "save_strategy": "no",
        "remove_unused_columns": False,
        "prediction_loss_only": True,  # evaluation keeps the loss only, never the logits
        "report_to": ["wandb"] if os.environ.get("WANDB_API_KEY") else [],
        "run_name": cfg["name"],
        **t,
    }


CE_CHUNK = 1024  # target positions scored at a time: float32 logits of 1024 x 256K ~ 1 GB


def target_loss(model: Any, inputs: dict[str, Any], num_items_in_batch: Any = None) -> Any:
    """Cross-entropy over the target tokens only, scoring just the positions that predict them.

    Position ``p`` predicts token ``p + 1``, so the model is asked for the logits at the
    positions just before each target (``logits_to_keep``, a tensor of indices). Batches of one
    conversation (the config's batch size). Summed and divided by ``num_items_in_batch`` (the
    targets across the accumulated batch) when the Trainer gives it, else averaged."""
    import torch

    labels = inputs["labels"]
    if labels.shape[0] != 1:
        raise ValueError("target_loss handles one conversation per batch (per_device_train_batch_size: 1)")
    nxt = labels[0, 1:]
    keep = torch.nonzero(nxt != -100).squeeze(-1)  # positions whose next token is a target
    out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep)
    logits, targets = out.logits[0], nxt[keep]
    total = logits.new_zeros((), dtype=torch.float32)
    for i in range(0, len(keep), CE_CHUNK):
        total = total + torch.nn.functional.cross_entropy(
            logits[i : i + CE_CHUNK].float(), targets[i : i + CE_CHUNK], reduction="sum"
        )
    if num_items_in_batch is not None:
        return total / num_items_in_batch
    return total / max(len(keep), 1)


def build_trainer(  # noqa: PLR0913 - the pieces a Trainer is made of
    model: Any,
    pad_id: int,
    train_enc: list[dict[str, list[int]]],
    held_enc: list[dict[str, list[int]]],
    cfg: dict[str, Any],
    out: Path,
    log: Any,
    bf16: bool,
) -> Any:
    """A Hugging Face Trainer over the tokenised examples; logs every step to ``log`` (JSON Lines).

    The loss is computed on the target positions only (`target_loss`): Aya's vocabulary is
    ~256K tokens, so the full logits of a 13K-token conversation are ~3.5 billion numbers, and
    the cross-entropy's float32 copy of them alone ran the H100 out of memory."""
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class TargetsOnly(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: ANN001, ANN202
            loss = target_loss(model, inputs, num_items_in_batch)
            return (loss, {"loss": loss}) if return_outputs else loss

    def collate(batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        return {k: torch.tensor(v) for k, v in pad_batch(batch, pad_id).items()}

    class Log(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw) -> None:  # noqa: ANN001, ANN003
            if logs:
                log.write(json.dumps({"step": state.global_step, **logs}) + "\n")
                log.flush()

    return TargetsOnly(
        model=model,
        args=TrainingArguments(**trainer_args(cfg, out, bool(held_enc), bf16)),
        train_dataset=train_enc,
        eval_dataset=held_enc or None,
        data_collator=collate,
        callbacks=[Log()],
    )


def main(argv: list[str] | None = None) -> int:  # noqa: C901, PLR0915 - one linear script
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path, nargs="?", default=CONFIG, help=f"default: {shown(CONFIG)}")
    p.add_argument("--dry-run", action="store_true", help="tokenise and report only (no GPU)")
    p.add_argument("--force", action="store_true", help="replace a finished run")
    p.add_argument("--push", action="store_true", help=f"upload the adapter to {HUB_ORG}/<name>, private")
    # a heal cycle (SEL-68, scripts/heal/train_and_serve.sh) passes its own fixes and folder
    p.add_argument("--data", type=Path, help="train on this JSONL instead of the config's data")
    p.add_argument("--out", type=Path, help="write the adapter here instead of runs/<name>")
    p.add_argument("--name", help="the run's name instead of the config's (summary, W&B, the Hub)")
    p.add_argument(
        "--init-adapter",
        type=Path,
        help="continue training this LoRA adapter (its own rank and modules) instead of a new one",
    )
    a = p.parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

    cfg = load(a.config)
    if a.data is not None:
        cfg["data"] = str(a.data.resolve())
    if not cfg["data"]:
        print(f"⛔ {shown(a.config)} names no data: pass --data")
        return 1
    if a.name:
        cfg["name"] = a.name
    if a.init_adapter is not None and not (a.init_adapter / "adapter_config.json").exists():
        print(f"⛔ no adapter to start from in {shown(a.init_adapter)} (no adapter_config.json)")
        return 1
    out = a.out.resolve() if a.out is not None else ROOT / "runs" / cfg["name"]
    if (out / "adapter_config.json").exists() and not a.force:
        print(f"⛔ {shown(out)} already has an adapter: pass --force to train again")
        return 1
    train_rows, held_rows = examples(cfg)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    tok.chat_template = relax_chat_template(tok.chat_template)  # the agent speaks first: see reflexive.sft
    lo = cfg.get("min_tokens", 64)  # fewer means the tokenizer's output isn't being read right
    train_enc, train_stats = encode_all(train_rows, tok, cfg["max_len"], lo)
    held_enc, held_stats = encode_all(held_rows, tok, cfg["max_len"], lo)
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
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    q = cfg.get("quantization", {})
    model = AutoModelForCausalLM.from_pretrained(
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
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lc = cfg["lora"]
    if a.init_adapter is not None:  # the next heal cycle starts from the adapter now served
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(a.init_adapter), is_trainable=True)
        lc = {**lc, "init_adapter": str(a.init_adapter)}
    else:
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
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("w")
    trainer = build_trainer(model, pad_id, train_enc, held_enc, cfg, out, log, bf16=True)
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
        "code": git_state(),  # the commit trained from, and uncommitted changes (SEL-22 provenance)
    }
    repo = f"{HUB_ORG}/{cfg['name']}"
    if a.push:
        summary["hub"] = repo
    # written before the upload, so a failed upload still leaves it (SEL-22: the provenance needs it)
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    if a.push:
        from huggingface_hub import HfApi

        model.push_to_hub(repo, private=True)
        for f in ("run_summary.json", "log.jsonl"):
            HfApi().upload_file(path_or_fileobj=str(out / f), path_in_repo=f, repo_id=repo)
    print(
        f"done in {minutes:.0f} min: train loss {result.training_loss:.3f}; held-out loss "
        f"{summary['held_out_loss']['before']} -> {summary['held_out_loss']['after']}; "
        f"peak {summary['peak_gpu_gb']} GB; adapter in {shown(out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
