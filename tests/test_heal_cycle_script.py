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


def test_a_sample_is_random_per_domain_but_the_same_for_a_seed():
    from reflexive.splits import task_ids

    a = cycle.heal_tasks(["telecom", "retail"], sample=3)
    assert a == cycle.heal_tasks(["telecom", "retail"], sample=3, seed=0)
    assert [d for d, _ in a] == ["telecom"] * 3 + ["retail"] * 3
    tel = [t for d, t in a if d == "telecom"]
    assert tel != task_ids("telecom", "train")[:3]  # not just the first ones
    assert tel == [t for t in task_ids("telecom", "train") if t in tel]  # the split's order
    assert a != cycle.heal_tasks(["telecom", "retail"], sample=3, seed=1)
    assert len(cycle.heal_tasks(["airline"], sample=500)) == len(task_ids("airline", "train"))
    with pytest.raises(ValueError, match="not both"):
        cycle.heal_tasks(["airline"], limit=2, sample=2)


def test_validation_tasks_are_never_healed_on(tmp_path):
    from reflexive.splits import task_ids

    hard = tmp_path / "hard_subset.jsonl"
    rows = [
        {"domain": "airline", "task_id": task_ids("airline", "train")[0], "group": "solved_by_aya"},
        {"domain": "airline", "task_id": task_ids("airline", "train")[1], "group": "teacher_unsolved"},
        {"domain": "airline", "task_id": task_ids("airline", "train")[2], "group": "healable"},
    ]
    hard.write_text("".join(json.dumps(r) + "\n" for r in rows))
    val = cycle.validation_tasks((tmp_path / "missing.jsonl", hard))
    assert val == {("airline", str(r["task_id"])) for r in rows[:2]}  # healable isn't validation
    everything = cycle.heal_tasks(["airline"], sample=500, exclude=val)
    assert len(everything) == len(task_ids("airline", "train")) - 2 and not set(everything) & val
    with pytest.raises(FileNotFoundError, match="--keep-validation"):
        cycle.validation_tasks((tmp_path / "missing.jsonl",))


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
