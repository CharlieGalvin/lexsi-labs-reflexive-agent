"""scripts/tau2/run_report.py: per-message cost summing and the token-based fallback."""

from __future__ import annotations

import json
import sys

import pytest
from _scripts import load_script

tau2_report = load_script("tau2/run_report")


def _run(tmp_path, monkeypatch, capsys, messages):
    f = tmp_path / "results.json"
    sim = {"task_id": "t1", "reward_info": {"reward": 1.0}, "agent_cost": None, "messages": messages}
    f.write_text(json.dumps({"simulations": [sim]}))
    monkeypatch.setattr(sys, "argv", ["tau2_report.py", str(f)])
    tau2_report.main()
    return capsys.readouterr().out


def test_sums_message_costs_and_ignores_uncosted_greeting(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, [
        {"role": "assistant", "content": "Hi! How can I help you today?", "cost": None},
        {"role": "user", "cost": 0.02, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"role": "assistant", "cost": 0.011, "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        {"role": "tool"},
    ])
    assert "agent:" in out and "$0.0110" in out
    assert "$0.0200" in out
    assert "total: $0.0310 for 1 task(s)" in out


def test_estimates_from_tokens_when_no_message_has_a_cost(tmp_path, monkeypatch, capsys):
    out = _run(tmp_path, monkeypatch, capsys, [
        {"role": "assistant", "usage": {"prompt_tokens": 1000, "completion_tokens": 100}},
    ])
    assert "~$0.0028 (estimated)" in out


def test_missing_run_exits_with_where_it_looked(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tau2_report.py", "no-such-run"])
    with pytest.raises(SystemExit, match="No results found"):
        tau2_report.main()
