"""SEL-65: the SFT trainer really trains, with whatever transformers and peft are installed.

A tiny random Llama with LoRA on the same modules as the config, two optimiser steps on CPU,
held-out loss before and after, the step log and the saved adapter. Skipped without transformers
and peft (they're in the `train` extra: the GPU pod has them). The 4-bit loading and the paged
8-bit optimizer need a GPU, so the optimizer here is plain AdamW.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from _scripts import load_script

from reflexive.files import ROOT

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

ts = load_script("sft/train_sft")


def _example(n: int, k: int) -> dict[str, list[int]]:
    ids = [(i * 7) % 60 + 2 for i in range(n)]
    return {"input_ids": ids, "labels": [-100] * (n - k) + ids[n - k :]}


def test_the_trainer_takes_two_steps_evaluates_logs_and_saves_an_adapter(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "sft_aya32b.yaml").read_text())
    cfg["trainer"].update(
        optim="adamw_torch", eval_steps=1, num_train_epochs=1, gradient_accumulation_steps=2
    )
    small = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = transformers.AutoModelForCausalLM.from_config(small)
    model.enable_input_require_grads()
    lc = cfg["lora"]
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=lc["r"],
            lora_alpha=lc["alpha"],
            lora_dropout=lc["dropout"],
            target_modules=lc["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    train = [_example(40, 10), _example(30, 5), _example(50, 20), _example(25, 8)]
    held = [_example(35, 6), _example(20, 4)]
    with (tmp_path / "log.jsonl").open("w") as log:
        trainer = ts.build_trainer(model, 0, train, held, cfg, tmp_path, log, bf16=False)
        before = trainer.evaluate()["eval_loss"]
        result = trainer.train()
        after = trainer.evaluate()["eval_loss"]
    model.save_pretrained(tmp_path)
    assert result.global_step == 2 and result.training_loss > 0 and before > 0 and after > 0
    logged = [json.loads(x) for x in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert any("loss" in r for r in logged) and any("eval_loss" in r for r in logged)
    assert (tmp_path / "adapter_config.json").exists()


def test_warmup_ratio_becomes_warmup_steps_where_transformers_dropped_it(tmp_path: Path) -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "sft_aya32b.yaml").read_text())
    kw = ts.trainer_args(cfg, tmp_path, has_held_out=True, bf16=False)
    transformers.TrainingArguments(**{**kw, "optim": "adamw_torch"})  # every keyword is accepted
    assert kw.get("warmup_ratio", kw.get("warmup_steps")) == 0.05
