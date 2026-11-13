"""The adapter lifecycle (SEL-22): the registry, the switch history and what's published."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from reflexive import adapters as ad

ROOT = Path(__file__).parents[1]


def test_the_real_registry_is_consistent():
    reg, state = ad.load_registry(), ad.load_state()
    assert ad.check_registry(reg) == [] and ad.check_state(reg, state) == []
    # only the state names Arm B, and it's an adapter that may be Arm B
    assert all(a["status"] != "current" for a in reg["adapters"])
    assert ad.adapters_by_id(reg)[state["current"]]["status"] in ad.SWITCHABLE


def test_check_registry_names_each_problem(tmp_path):
    reg = {
        "base": "b",
        "hub_org": "org",
        "adapters": [
            {"id": "a", "repo": "r", "status": "adopted", "parent": "ghost", "train_config": "nope.yaml"},
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
    summary["code"] = {"commit": "5f1b5b5", "dirty": False}
    prov = ad.provenance(reg, v2, summary, {"commit": "a2da3d7", "dirty": False}, status="current")
    assert prov["format"] == ad.PROVENANCE_FORMAT
    assert prov["hub_repo"] == "reagent-lexsi/sft-aya32b-r16-v2"
    assert prov["base_model"] == "CohereLabs/aya-expanse-32b"
    assert prov["code_commit"] == "5f1b5b5"  # the training commit, not where it was published from
    assert prov["published_from"]["commit"] == "a2da3d7"
    assert prov["training"]["data"]["sha256"] == "f00d"
    json.dumps(prov)  # serialisable as written
    card = ad.model_card(reg, v2, prov)
    assert card.startswith("---\nbase_model: CohereLabs/aya-expanse-32b")
    assert "**Arm B**" in card and "17.0 / 30" in card and "0.854 → 0.368" in card
    assert "mean of 2; samples 16, 18" in card and "trained from code commit `5f1b5b5`" in card
    assert 'ADAPTER= EXTRA_ADAPTERS="sft2=runs/hub/sft-aya32b-r16-v2"' in card


def test_a_card_without_a_run_summary_says_so():
    reg = ad.load_registry()
    v4 = ad.adapters_by_id(reg)["sft-v4"]
    card = ad.model_card(reg, v4, ad.provenance(reg, v4, None, None))
    assert "wasn't kept" in card and "Not adopted" in card and "wasn't recorded" in card


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
        sys, "argv", ["use.py", "sft-v1", "--why", "a drill", "--no-download", "--by", "Charlie"]
    )
    assert use.main() == 0
    assert saved["current"] == "sft-v1" and saved["history"][-1]["from"] == "sft-v2"
    assert saved["history"][-1]["by"] == "Charlie"


def test_use_refuses_a_broken_registry(monkeypatch, capsys):
    use = _script("use")
    monkeypatch.setattr(ad, "check_registry", lambda reg, root=None: ["an id appears twice"])
    monkeypatch.setattr(sys, "argv", ["use.py", "sft-v2", "--why", "test", "--no-download"])
    assert use.main() == 1
    assert "fix configs/adapters.yaml" in capsys.readouterr().out


def test_publish_dry_run_writes_the_card_and_provenance(tmp_path, monkeypatch):
    pub = _script("publish")
    monkeypatch.setattr(pub, "ROOT", tmp_path)
    (tmp_path / "results").mkdir()
    stale = tmp_path / "results" / "adapters" / "sft-v4" / "eval" / "old.metrics.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")  # left from an earlier publish
    monkeypatch.setattr(sys, "argv", ["publish.py", "sft-v4", "--dry-run"])
    monkeypatch.setattr(ad, "check_registry", lambda reg, root=None: [])
    assert pub.main() == 0
    out = tmp_path / "results" / "adapters" / "sft-v4"
    assert (out / "README.md").read_text().startswith("---")
    assert json.loads((out / "lexsi_provenance.json").read_text())["id"] == "sft-v4"
    assert not stale.exists()  # eval/ holds only this publish's metrics files


# --- review fixes -----------------------------------------------------------------------------


def test_the_state_not_the_registry_decides_arm_b():
    v2 = {"id": "sft-v2", "status": "adopted"}
    first = {"current": "sft-v2", "history": [{"from": None, "to": "sft-v2"}]}
    assert ad.status_of(v2, first) == "current"
    later = ad.switch(first, "grpo-v1", "better", now="t")
    assert ad.status_of(v2, later) == "superseded"  # it was Arm B and was switched away from
    assert ad.status_of({"id": "sft-v4", "status": "not_adopted"}, later) == "not_adopted"


def test_check_state_names_an_unknown_adapter():
    reg = {"adapters": [{"id": "a"}]}
    assert ad.check_state(reg, {"current": "ghost", "history": []}) == [
        "state.json's current adapter 'ghost' isn't in the registry"
    ]


def test_a_card_without_held_out_loss_says_so():
    s = {"train_loss": 0.4, "held_out_loss": {"before": None, "after": None}}
    assert ad._loss_line(s) == "- Train loss 0.400; no held-out set"


def test_eval_rows_use_the_entrys_own_counts():
    rows = ad._eval_lines({"val": {"mean": 9.0, "of": 20, "samples": [8, 9, 10]}})
    assert rows == ["| Validation (unseen train tasks), 20 tasks | 9.0 / 20 | mean of 3; samples 8, 9, 10 |"]


def _use(monkeypatch, argv, state):
    use = _script("use")
    monkeypatch.setattr(ad, "load_state", lambda: state)
    saved = {}
    monkeypatch.setattr(ad, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(sys, "argv", ["use.py", *argv])
    return use, saved


def test_use_refuses_the_adapter_already_current(monkeypatch, capsys):
    use, saved = _use(
        monkeypatch, ["sft-v2", "--why", "x", "--no-download"], {"current": "sft-v2", "history": []}
    )
    assert use.main() == 1 and not saved
    assert "already current" in capsys.readouterr().out


def test_use_refuses_a_control_or_rejected_adapter_without_force(monkeypatch, capsys):
    state = {"current": "sft-v2", "history": []}
    use, saved = _use(monkeypatch, ["sft-long", "--why", "x", "--no-download"], state)
    assert use.main() == 1 and not saved and "control" in capsys.readouterr().out
    use, saved = _use(monkeypatch, ["sft-v3", "--why", "a drill", "--no-download", "--force"], state)
    assert use.main() == 0 and saved["current"] == "sft-v3"


class Reply:
    def __init__(self, code, text=""):
        self.status_code, self.text = code, text


def _server(monkeypatch, replies):
    use = _script("use")
    calls = []

    def post(server, what, body):
        calls.append((what, body["lora_name"], body.get("lora_path")))
        return replies.pop(0)

    monkeypatch.setattr(use, "_post", post)
    return use, calls


def test_a_bad_adapter_is_refused_before_current_is_touched(monkeypatch):
    use, calls = _server(monkeypatch, [Reply(400, "rank 32 > max_lora_rank 16")])
    with pytest.raises(SystemExit, match="unchanged"):
        use.load_into_server("http://x", "new", "old")
    assert calls == [("load", "current-check", "new")]  # current never unloaded


def test_a_failed_swap_loads_the_old_adapter_back(monkeypatch):
    replies = [Reply(200), Reply(200), Reply(200), Reply(500, "boom"), Reply(200)]
    use, calls = _server(monkeypatch, replies)
    with pytest.raises(SystemExit, match="loaded back"):
        use.load_into_server("http://x", "new", "old")
    assert calls[-1] == ("load", "current", "old")


def test_an_unload_refusal_other_than_not_loaded_stops_the_swap(monkeypatch):
    replies = [Reply(200), Reply(200), Reply(500, "server busy")]
    use, calls = _server(monkeypatch, replies)
    with pytest.raises(SystemExit, match="refused to unload"):
        use.load_into_server("http://x", "new", "old")
    replies = [Reply(200), Reply(200), Reply(400, "The lora adapter 'current' cannot be found."), Reply(200)]
    use, calls = _server(monkeypatch, replies)
    use.load_into_server("http://x", "new", None)  # nothing loaded yet: fine
    assert calls[-1] == ("load", "current", "new")
