"""SEL-68: `train_sft.py`'s options for a heal cycle, and the heal config. No GPU, no model:
these stop before the tokenizer is loaded."""

from __future__ import annotations

from pathlib import Path

import yaml
from _scripts import load_script

from reflexive.files import ROOT

ts = load_script("sft/train_sft")
HEAL = ROOT / "configs" / "train" / "sft_heal.yaml"


def test_the_heal_config_is_v1_but_for_its_data_group_and_steps():
    heal, v1 = (yaml.safe_load(p.read_text()) for p in (HEAL, ROOT / "configs/train/sft_aya32b.yaml"))
    assert heal["data"] is None and heal["train_group"] == "heal"
    assert (heal["model"], heal["lora"], heal["max_len"]) == (v1["model"], v1["lora"], v1["max_len"])
    same = {k: v for k, v in v1["trainer"].items() if k not in ("gradient_accumulation_steps", "eval_steps")}
    assert {k: heal["trainer"][k] for k in same} == same
    assert heal["trainer"]["gradient_accumulation_steps"] == 2


def test_the_heal_config_needs_data(capsys):
    assert ts.main([str(HEAL)]) == 1
    assert "pass --data" in capsys.readouterr().out


def test_a_starting_adapter_must_exist(tmp_path: Path, capsys):
    fixes = tmp_path / "fixes.jsonl"
    fixes.write_text("")
    code = ts.main(
        [str(HEAL), "--data", str(fixes), "--out", str(tmp_path / "o"), "--init-adapter", str(tmp_path)]
    )
    assert code == 1 and "no adapter to start from" in capsys.readouterr().out


def test_a_finished_adapter_in_out_is_not_overwritten(tmp_path: Path, capsys):
    fixes, out = tmp_path / "fixes.jsonl", tmp_path / "heal-c1"
    fixes.write_text("")
    out.mkdir()
    (out / "adapter_config.json").write_text("{}")
    assert ts.main([str(HEAL), "--data", str(fixes), "--out", str(out), "--name", "heal-c1"]) == 1
    assert "already has an adapter" in capsys.readouterr().out
