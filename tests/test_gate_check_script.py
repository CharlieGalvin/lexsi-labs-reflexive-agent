"""`just gate-check` in check mode: it counts the recorded heal cycles but never adds one."""

import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

from reflexive.gate import GateDecision

spec = importlib.util.spec_from_file_location(
    "gate_check", Path(__file__).parents[1] / "scripts/arms/gate_check.py"
)
assert spec is not None and spec.loader is not None
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)


def test_a_check_respects_the_budget_and_leaves_the_history_alone(tmp_path, monkeypatch, capsys):
    history = tmp_path / "gate_history.jsonl"
    used = [GateDecision(True, 0.1, 0.5, "ok", "promote", "all") for _ in range(3)]
    history.write_text("".join(json.dumps(asdict(d)) + "\n" for d in used))
    monkeypatch.setattr(gc, "HISTORY", history)
    monkeypatch.setattr(gc, "dev_run", lambda arm, sample: [])
    monkeypatch.setattr(sys, "argv", ["gate_check.py", "P", "S2"])

    gc.main()

    out = capsys.readouterr().out
    assert "heal cycles gated so far: 3 of 3" in out
    assert "GATE: NO_DECISION (decided by: budget)" in out
    assert len(history.read_text().splitlines()) == 3  # nothing appended
    assert json.loads((tmp_path / "gate-P-vs-S2-s0.json").read_text())["check"] == "budget"
