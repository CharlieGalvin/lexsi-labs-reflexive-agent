"""The adapter lifecycle (SEL-22): the registry, the switch history and what's published."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from reflexive import adapters as ad

ROOT = Path(__file__).parents[1]


def test_the_real_registry_is_consistent():
    reg = ad.load_registry()
    assert ad.check_registry(reg) == []
    current = [a["id"] for a in reg["adapters"] if a["status"] == "current"]
    assert current == [ad.load_state()["current"]]  # registry and state agree on Arm B


def test_check_registry_names_each_problem(tmp_path):
    reg = {
        "base": "b",
        "hub_org": "org",
        "adapters": [
            {"id": "a", "repo": "r", "status": "current", "parent": "ghost", "train_config": "nope.yaml"},
            {"id": "a", "repo": "r", "status": "maybe"},
        ],
    }
    problems = ad.check_registry(reg, root=tmp_path)
    assert "an id appears twice" in problems
    assert "a Hub repo appears twice" in problems
    assert any("parent 'ghost'" in p for p in problems)
    assert any("nope.yaml doesn't exist" in p for p in problems)
    assert any("status 'maybe'" in p for p in problems)


def test_switch_records_history_and_rollback_returns_to_the_one_before():
    s0 = {"current": "sft-v2", "revisions": {"grpo-v1": "abc123"}, "history": []}
    s1 = ad.switch(s0, "grpo-v1", "beat v2 on validation", by="katie", now="2026-11-12T10:00:00+00:00")
    assert s0["current"] == "sft-v2" and s0["history"] == []  # the old state is unchanged
    assert s1["current"] == "grpo-v1"
    assert s1["history"][-1] == {
        "at": "2026-11-12T10:00:00+00:00",
        "from": "sft-v2",
        "to": "grpo-v1",
        "revision": "abc123",
        "why": "beat v2 on validation",
        "by": "katie",
    }
    assert ad.previous(s1) == "sft-v2"
    assert ad.previous({"current": None, "history": []}) is None


def test_provenance_and_card_from_a_run_summary():
    reg = ad.load_registry()
    v2 = ad.adapters_by_id(reg)["sft-v2"]
    summary = {
        "steps": 74,
        "train_loss": 0.426,
        "held_out_loss": {"before": 0.854, "after": 0.368},
        "lora": {"r": 16, "lora_alpha": 32},
        "trainer": {"learning_rate": 1e-4},
        "minutes": 31.0,
        "gpu": "NVIDIA H100 80GB HBM3",
        "peak_gpu_gb": 54.0,
        "data": {"file": "data/train/sft-v2.jsonl", "sha256": "f00d"},
    }
    prov = ad.provenance(reg, v2, summary, "5f1b5b5")
    assert prov["format"] == ad.PROVENANCE_FORMAT
    assert prov["hub_repo"] == "reagent-lexsi/sft-aya32b-r16-v2"
    assert prov["base_model"] == "CohereLabs/aya-expanse-32b" and prov["code_commit"] == "5f1b5b5"
    assert prov["training"]["data"]["sha256"] == "f00d"
    json.dumps(prov)  # serialisable as written
    card = ad.model_card(reg, v2, prov)
    assert card.startswith("---\nbase_model: CohereLabs/aya-expanse-32b")
    assert "**Arm B**" in card and "17.0 / 30" in card and "0.854 → 0.368" in card
    assert 'EXTRA_ADAPTERS="sft2=runs/sft-aya32b-r16-v2"' in card


def test_a_card_without_a_run_summary_says_so():
    reg = ad.load_registry()
    v4 = ad.adapters_by_id(reg)["sft-v4"]
    card = ad.model_card(reg, v4, ad.provenance(reg, v4, None, None))
    assert "wasn't kept" in card and "Not adopted" in card


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "adapters" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_rollback_with_nothing_to_go_back_to_refuses(tmp_path, monkeypatch, capsys):
    use = _script("use")
    monkeypatch.setattr(ad, "load_state", lambda: {"current": "sft-v2", "revisions": {}, "history": []})
    monkeypatch.setattr(sys, "argv", ["use.py", "--rollback", "--why", "test"])
    assert use.main() == 1
    assert "nothing to roll back to" in capsys.readouterr().out


def test_use_records_the_switch_without_downloading(tmp_path, monkeypatch):
    use = _script("use")
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(ad, "STATE", state_file)
    monkeypatch.setattr(ad, "load_state", lambda: {"current": "sft-v2", "revisions": {}, "history": []})
    saved = {}
    monkeypatch.setattr(ad, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(
        sys, "argv", ["use.py", "sft-v3", "--why", "a drill", "--no-download", "--by", "Charlie"]
    )
    assert use.main() == 0
    assert saved["current"] == "sft-v3" and saved["history"][-1]["from"] == "sft-v2"
    assert saved["history"][-1]["by"] == "Charlie"


def test_use_refuses_a_broken_registry(monkeypatch, capsys):
    use = _script("use")
    monkeypatch.setattr(
        ad, "check_registry", lambda reg: ["exactly one adapter must have status current (Arm B)"]
    )
    monkeypatch.setattr(sys, "argv", ["use.py", "sft-v2", "--why", "test", "--no-download"])
    assert use.main() == 1
    assert "fix configs/adapters.yaml" in capsys.readouterr().out


def test_publish_dry_run_writes_the_card_and_provenance(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    pub = _script("publish")
    monkeypatch.setattr(pub, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(sys, "argv", ["publish.py", "sft-v4", "--dry-run"])
    monkeypatch.setattr(ad, "check_registry", lambda reg: [])
    assert pub.main() == 0
    out = tmp_path / "results" / "adapters" / "sft-v4"
    assert (out / "README.md").read_text().startswith("---")
    assert json.loads((out / "lexsi_provenance.json").read_text())["id"] == "sft-v4"
