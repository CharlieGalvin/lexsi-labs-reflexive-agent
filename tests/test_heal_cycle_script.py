"""SEL-47: scripts/heal/cycle.py's task choice, student factory and adapter loading (no API)."""

from __future__ import annotations

import json
import sys
import types

import pytest
from _scripts import load_script

cycle = load_script("heal/cycle")


def test_the_gate_gets_every_dev_task_and_healing_only_train_tasks():
    from reflexive.splits import split_of

    dev = cycle.gate_tasks()
    assert len(dev) == 36 and {split_of(*t) for t in dev} == {"dev"}
    heal = cycle.heal_tasks(["telecom", "airline"], 2)
    assert len(heal) == 4 and {split_of(*t) for t in heal} == {"train"}


def test_the_cohere_student_refuses_an_adapter(monkeypatch):
    made = []
    monkeypatch.setattr(cycle, "make_student", lambda s: made.append(s) or s)
    arm = types.SimpleNamespace(
        student={"api_base": "https://api.cohere.ai/compatibility/v1", "model": "aya"}
    )
    make = cycle.student_factory(arm)
    assert make(None)["model"] == "aya"
    with pytest.raises(RuntimeError, match="vLLM"):
        make("heal-1")


def test_a_served_student_runs_the_adapter_as_its_model(monkeypatch):
    monkeypatch.setattr(cycle, "make_student", lambda s: s)
    arm = types.SimpleNamespace(student={"api_base": "http://localhost:8000/v1", "model": "base"})
    assert cycle.student_factory(arm)("heal-1")["model"] == "heal-1"


def test_a_new_adapter_is_registered_with_vllm(tmp_path, monkeypatch):
    sent = []

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(cycle.urllib.request, "urlopen", lambda req, timeout: sent.append(req) or Resp())
    write = "import pathlib, sys; pathlib.Path(sys.argv[1], 'adapter_config.json').write_text('{}')"
    trainer = cycle.LoadingTrainer(
        tmp_path, command=[sys.executable, "-c", write, "{out}"], api_base="http://h:8000/v1/"
    )
    name = trainer.train([{"format": "reagent.sft/v1", "id": "x"}], None)

    (req,) = sent
    assert req.full_url == "http://h:8000/v1/load_lora_adapter"
    assert json.loads(req.data) == {"lora_name": name, "lora_path": str((tmp_path / name).resolve())}


def test_no_adapter_written_means_no_swap(tmp_path):
    trainer = cycle.LoadingTrainer(tmp_path, command=[sys.executable, "-c", "pass"], api_base="http://h/v1")
    with pytest.raises(RuntimeError, match="no adapter"):
        trainer.train([{"format": "reagent.sft/v1", "id": "x"}], None)
