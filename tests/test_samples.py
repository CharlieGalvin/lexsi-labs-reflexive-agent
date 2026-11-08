"""`just arm-samples`: passes per domain and sample, re-runs replacing their crash."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/arms/samples.py"
spec = importlib.util.spec_from_file_location("samples", SCRIPT)
assert spec is not None and spec.loader is not None
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)


def test_counts_per_sample_and_reruns(tmp_path, monkeypatch):
    arms = tmp_path / "results" / "episodes" / "arms"
    arms.mkdir(parents=True)
    rows = [
        {"task_id": "1", "sample": 0, "outcome": {"passed": True}},
        {"task_id": "1", "sample": 1, "outcome": {"passed": False}, "error": "Connection error"},
        {"task_id": "1", "sample": 1, "outcome": {"passed": True}},  # the re-run replaces the crash
        {"task_id": "2", "sample": 0, "outcome": {"passed": False}},
    ]
    (arms / "S-retail-dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(sm, "ROOT", tmp_path)
    t = sm.table("S", "dev")
    assert t[("retail", 0)][:2] == [1, 2]
    assert t[("retail", 1)] == [1, 1, 0]
