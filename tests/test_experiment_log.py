import argparse
import csv
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("experiment_log", ROOT / "scripts" / "experiment_log.py")
assert spec is not None and spec.loader is not None
el = importlib.util.module_from_spec(spec)
spec.loader.exec_module(el)


def _rows():
    # 4 questions: 2 correct; one badly normalised ID (an error), one wrong customer
    base = {"customer_id": "C1000", "field": "plan", "format": "1.0"}
    return [
        {
            **base,
            "level": "clean",
            "correct": "1.0",
            "tool_ok": "1.0",
            "n_tool_calls": "1",
            "hallucinated_ids": "0",
            "wrong_customer": "0",
        },
        {
            **base,
            "level": "clean",
            "correct": "1.0",
            "tool_ok": "1.0",
            "n_tool_calls": "1",
            "hallucinated_ids": "0",
            "wrong_customer": "0",
        },
        {
            **base,
            "level": "lower_dash",
            "correct": "0.0",
            "tool_ok": "0.0",
            "n_tool_calls": "1",
            "hallucinated_ids": "1",
            "wrong_customer": "0",
        },
        {
            **base,
            "level": "distractor",
            "correct": "0.0",
            "tool_ok": "0.0",
            "n_tool_calls": "1",
            "hallucinated_ids": "0",
            "wrong_customer": "1",
        },
    ]


def _metrics(run="base", **kw):
    args = dict(
        run=run,
        arm="A",
        config="configs/train/messy_grpo.yaml",
        model="CohereLabs/tiny-aya-global",
        adapter=None,
        commit="abc1234",
        owner="tester",
        eval_csv="runs/base-messy/eval_eval.csv",
    )
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


LOG_WITH_PLACEHOLDER = """# Experiment log

| Date | Run | Owner | Config | Change | task_success | calls/success | halluc. rate | Notes |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-11-0? | base | | | untuned baseline (Arm A) | | | | |
| 2026-11-04 | arm-A | Chiara | x | y | 5/74 | 242 | 98.6% | z |
"""


def test_row_fills_an_empty_placeholder_row():
    row = el.log_row(_metrics(), "untuned baseline (Arm A)")
    new = el.add_to_log(LOG_WITH_PLACEHOLDER, "base", row)
    lines = new.splitlines()
    assert len(lines) == len(LOG_WITH_PLACEHOLDER.splitlines())  # filled, not added
    assert lines[4] == row and "| 2026-11-0? | base |" not in new
    assert lines[5].startswith("| 2026-11-04 | arm-A |")  # other rows untouched


def test_row_is_appended_to_the_runs_table_not_the_curator_table():
    # the real log has two tables: training/eval runs (task_success), then Curator data runs
    text = (ROOT / "docs" / "EXPERIMENTS.md").read_text()
    row = el.log_row(_metrics("messy-grpo-2"), "GRPO 100 steps")
    lines = el.add_to_log(text, "messy-grpo-2", row).splitlines()
    assert len(lines) == len(text.splitlines()) + 1
    at = lines.index(row)
    assert lines[at - 1].startswith("|") and not lines[at + 1].startswith("|")  # end of a table
    assert at < next(i for i, line in enumerate(lines) if line.startswith("## Data runs"))


def test_row_cells_and_links():
    m = _metrics(
        "messy-grpo",
        training={"steps": 100, "wandb_url": "https://wandb.ai/x/y/runs/z", "hub_repo": "org/adapter"},
    )
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
    args = argparse.Namespace(
        eval_csv=eval_csv,
        run="base",
        arm="A",
        adapter=None,
        config="configs/train/messy_grpo.yaml",
        commit="abc1234",
        owner="tester",
        change=None,
        notes="",
    )
    assert el.record(args) == 0
    m = json.loads((tmp_path / "results" / "base.metrics.json").read_text())
    assert m["arm"] == "A" and m["task_success"] == 0.5
    assert "| base | tester |" in log.read_text()
    assert el.record(args) == 1  # never overwritten


@pytest.mark.parametrize("text", ["no table here", "| a | b |\n|---|---|\n| 1 | 2 |\n"])
def test_add_to_log_needs_a_runs_table(text):
    with pytest.raises(ValueError):
        el.add_to_log(text, "base", "| row |")


def test_compare_saves_the_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(el, "RESULTS", tmp_path)
    monkeypatch.setattr(el, "ROOT", tmp_path)
    for name in ("base", "after"):
        (tmp_path / f"{name}.metrics.json").write_text(json.dumps(_metrics(name)))
    assert el.compare(argparse.Namespace(run_a="base", run_b="after")) == 0
    saved = (tmp_path / "base-vs-after.compare.md").read_text()
    assert "| task_success | 50.0% | 50.0% | +0.0 pts |" in saved
    assert "| task_success |" in capsys.readouterr().out
