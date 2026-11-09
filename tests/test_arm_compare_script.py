"""SEL-27: scripts/arms/compare.py writes the table and the committed metrics (no API)."""

from __future__ import annotations

import json
import sys

import pytest
from _scripts import load_script
from test_compare import runs

cmp = load_script("arms/compare")


def write(path, recs):
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return path


def test_the_script_writes_the_table_and_the_metrics(tmp_path, monkeypatch, capsys):
    a, b = runs()
    fa, fb = write(tmp_path / "A-telecom-train.jsonl", a), write(tmp_path / "B-telecom-train.jsonl", b)
    monkeypatch.setattr(cmp, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(sys, "argv", ["compare.py", str(fa), str(fb), "--names", "A", "B", "--reps", "500"])
    assert cmp.main() == 0
    assert "| Success rate |" in capsys.readouterr().out
    m = json.loads((tmp_path / "results" / "A-vs-B.compare.metrics.json").read_text())
    assert m["a"]["name"] == "A" and m["reps"] == 500 and m["rows"][0]["metric"] == "success_rate"
    assert (tmp_path / "results" / "A-vs-B.compare.md").read_text().startswith("# A vs B")


def test_runs_with_no_task_in_common_are_refused(tmp_path, monkeypatch):
    a, b = runs()
    for r in b:
        r["task_id"] = "other-" + r["task_id"]
    fa, fb = write(tmp_path / "a.jsonl", a), write(tmp_path / "b.jsonl", b)
    monkeypatch.setattr(sys, "argv", ["compare.py", str(fa), str(fb), "--no-write"])
    with pytest.raises(SystemExit):
        cmp.main()
