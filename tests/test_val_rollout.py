"""SEL-49: an arm on the 30 validation tasks (`just arm-val`), and arm G's config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from _scripts import load_script

from reflexive.arms import get_arm
from reflexive.files import ROOT

vr = load_script("arms/val_rollout")


def _val(folder: Path, domain: str, *task_ids: str) -> None:
    rows = [{"task_id": t, "sample": s, "domain": domain} for t in task_ids for s in (0, 1)]
    (folder / f"P-{domain}-val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def source(tmp_path: Path) -> Path:
    _val(tmp_path, "retail", "3", "0")
    return tmp_path


def test_task_ids_come_from_arm_p_s_validation_file_once_each(source: Path):
    assert vr.val_tasks(source, "retail") == ["0", "3"]


def test_a_missing_validation_file_says_where_to_get_it(source: Path):
    with pytest.raises(SystemExit, match="copy P-\\*-val.jsonl"):
        vr.val_tasks(source, "airline")


@pytest.mark.parametrize("task", ["1", "5"])  # retail 1 is dev, 5 is test
def test_a_task_outside_train_is_refused(tmp_path: Path, task: str):
    _val(tmp_path, "retail", "0", task)
    with pytest.raises((SystemExit, ValueError)):
        vr.val_tasks(tmp_path, "retail")


def test_the_rollout_command_keeps_ids_whole_and_writes_the_val_file():
    ids = ["[mobile_data_issue]x|y", "2"]  # telecom-style IDs with | and [ ]
    cmd = vr.command("G", "telecom", ids, Path("configs/sft/arms.grpo.yaml"), 2, ["--workers", "8"])
    assert cmd[2:4] == ["G", "telecom"]
    assert cmd[cmd.index("--task-ids") + 1 : cmd.index("--samples")] == ids
    assert cmd[cmd.index("--samples") + 1] == "2"
    assert cmd[cmd.index("--out") + 1].endswith("results/episodes/arms/G-telecom-val.jsonl")
    assert cmd[-2:] == ["--workers", "8"]


def test_runs_each_domain_in_turn_and_stops_at_a_failure(source: Path, monkeypatch: pytest.MonkeyPatch):
    _val(source, "airline", "0")
    ran: list[list[str]] = []

    class Done:
        def __init__(self, code: int):
            self.returncode = code

    monkeypatch.setattr(vr, "trained_tasks", lambda: set())
    monkeypatch.setattr(vr.subprocess, "run", lambda cmd, check: (ran.append(cmd), Done(0))[1])
    argv = ["G", "c.yaml", "--from", str(source), "--domains", "retail", "airline"]
    assert vr.main(argv) == 0
    assert [c[3] for c in ran] == ["retail", "airline"]

    ran.clear()
    monkeypatch.setattr(vr.subprocess, "run", lambda cmd, check: (ran.append(cmd), Done(3))[1])
    assert vr.main(argv) == 3 and len(ran) == 1


def test_a_task_grpo_trained_on_is_refused(source: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(vr, "trained_tasks", lambda: {("retail", "3")})
    with pytest.raises(SystemExit, match="GRPO's training data"):
        vr.main(["G", "c.yaml", "--from", str(source), "--domains", "retail", "--dry-run"])


def test_dry_run_prints_without_running(source: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(vr, "trained_tasks", lambda: set())
    monkeypatch.setattr(vr.subprocess, "run", lambda *a, **k: pytest.fail("ran a rollout in a dry run"))
    assert vr.main(["G", "c.yaml", "--from", str(source), "--domains", "retail", "--dry-run"]) == 0
    assert "Arm G on validation: retail 2 tasks (2), 2 samples each" in capsys.readouterr().out


def _cfg(name: str) -> dict[str, Any]:
    return yaml.safe_load((ROOT / "configs" / "sft" / name).read_text())


def test_arm_g_is_s2_with_only_the_adapter_changed():
    g, s2 = _cfg("arms.grpo.yaml"), _cfg("arms.sft2.yaml")
    differ = {k for k in g["student"] if g["student"][k] != s2["student"].get(k)}
    assert differ == {"name", "model"} and g["student"]["model"] == "grpo"
    assert g["customer"] == s2["customer"]
    assert g["arms"]["G"]["prompt_variant"] == s2["arms"]["S2"]["prompt_variant"]
    arm = get_arm("G", ROOT / "configs" / "sft" / "arms.grpo.yaml")
    assert arm.student["api_base"] == "http://localhost:8000/v1"
