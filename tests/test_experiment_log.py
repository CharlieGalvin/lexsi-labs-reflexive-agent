import argparse
import csv
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_log", ROOT / "scripts" / "experiment_log.py")
el = importlib.util.module_from_spec(spec)
spec.loader.exec_module(el)


def _rows():
    # 4 questions: 2 correct; one badly normalised ID (an error), one wrong customer
    base = {"customer_id": "C1000", "field": "plan", "format": "1.0"}
    return [
        {**base, "level": "clean", "correct": "1.0", "tool_ok": "1.0", "n_tool_calls": "1",
         "hallucinated_ids": "0", "wrong_customer": "0"},
        {**base, "level": "clean", "correct": "1.0", "tool_ok": "1.0", "n_tool_calls": "1",
         "hallucinated_ids": "0", "wrong_customer": "0"},
        {**base, "level": "lower_dash", "correct": "0.0", "tool_ok": "0.0", "n_tool_calls": "1",
         "hallucinated_ids": "1", "wrong_customer": "0"},
        {**base, "level": "distractor", "correct": "0.0", "tool_ok": "0.0", "n_tool_calls": "1",
         "hallucinated_ids": "0", "wrong_customer": "1"},
    ]


def _metrics(run="base", **kw):
    args = dict(run=run, arm="A", config="configs/train/messy_grpo.yaml",
                model="CohereLabs/tiny-aya-global", adapter=None, commit="abc1234",
                owner="tester", eval_csv="runs/base-messy/eval_eval.csv")
    return el.metrics(_rows(), **{**args, **kw})


def test_wilson_interval():
    lo, hi = el.wilson(64, 120)
    assert 0.44 < lo < 0.45 and 0.62 < hi < 0.63
    assert el.wilson(0, 0) == (0.0, 0.0)


def test_metrics():
    m = _metrics()
    assert (m["passed"], m["questions"], m["task_success"]) == (2, 4, 0.5)
    assert m["hallucination_rate"] == 0.5 and m["hallucinated"] == 2
    assert m["calls_per_success"] == 2.0
    assert list(m["by_type"]) == ["clean", "lower_dash", "distractor"]
    assert m["by_type"]["clean"] == {"passed": 2, "n": 2, "tool_ok": 1.0, "hallucination_rate": 0.0}


def test_row_fills_the_base_placeholder_in_the_real_log():
    text = (ROOT / "docs" / "EXPERIMENTS.md").read_text()
    row = el.log_row(_metrics(), "untuned baseline (Arm A)")
    new = el.add_to_log(text, "base", row)
    assert row in new
    assert len(new.splitlines()) == len(text.splitlines())  # filled, not added
    assert "| 2026-11-0? | base |" not in new


def test_row_is_appended_when_there_is_no_placeholder():
    text = (ROOT / "docs" / "EXPERIMENTS.md").read_text()
    row = el.log_row(_metrics("messy-grpo"), "GRPO 100 steps")
    new = el.add_to_log(text, "messy-grpo", row)
    lines = new.splitlines()
    assert row in lines and len(lines) == len(text.splitlines()) + 1
    table = [line for line in lines if line.startswith("|")]
    assert table[-1] == row


def test_row_cells_and_links():
    m = _metrics("messy-grpo", training={"steps": 100, "wandb_url": "https://wandb.ai/x/y/runs/z",
                                         "hub_repo": "org/adapter"})
    row = el.log_row(m, "GRPO 100 steps")
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert len(cells) == 9
    assert cells[5].startswith("2/4 = 50.0%") and cells[6] == "2.0" and cells[7].startswith("50.0%")
    assert "[W&B](https://wandb.ai/x/y/runs/z)" in cells[8] and "huggingface.co/org/adapter" in cells[8]


def test_compare_table():
    a, b = _metrics("base"), _metrics("after")
    b["task_success"], b["by_type"]["clean"]["passed"] = 0.75, 1
    table = el.compare_table(a, b)
    assert "| task_success | 50.0% | 75.0% | +25.0 pts |" in table
    assert "| correct: `clean` | 2/2 | 1/2 | -50.0 pts |" in table


def test_record_writes_metrics_and_refuses_to_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "RESULTS", tmp_path / "results")
    log = tmp_path / "EXPERIMENTS.md"
    log.write_text((ROOT / "docs" / "EXPERIMENTS.md").read_text())
    monkeypatch.setattr(el, "LOG", log)
    monkeypatch.setattr(el, "ROOT", tmp_path)
    eval_csv = tmp_path / "eval_eval.csv"
    with open(eval_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(_rows()[0]))
        w.writeheader()
        w.writerows(_rows())
    args = argparse.Namespace(eval_csv=eval_csv, run="base", arm="A", adapter=None,
                              config="configs/train/messy_grpo.yaml", commit="abc1234",
                              owner="tester", change=None, notes="")
    assert el.record(args) == 0
    m = json.loads((tmp_path / "results" / "base.metrics.json").read_text())
    assert m["arm"] == "A" and m["task_success"] == 0.5
    assert "| base | tester |" in log.read_text()
    assert el.record(args) == 1  # never overwritten


@pytest.mark.parametrize("bad", ["| a | b |"])
def test_add_to_log_needs_a_table(bad):
    with pytest.raises(ValueError):
        el.add_to_log("no table here", "base", bad)
