"""SEL-110's conversations dataset (scripts/data/publish_conversations.py): checks and the dry run.

No network: the dry run writes data/conversations/ and uploads nothing.
"""

from __future__ import annotations

import json

import pytest
from _scripts import load_script

from reflexive.splits import task_ids

pc = load_script("data/publish_conversations")


def rec(task, passed=True, sample=0, crashed=False):
    r = {"task_id": task, "sample": sample, "termination": "user_stop", "outcome": {"passed": passed}}
    if crashed:
        r.update(termination="error", error="RateLimitError", outcome=None)
    return r


def write(d, name, recs):
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d / name


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "OUT", tmp_path / "out")
    return tmp_path / "out"


def test_counts_a_file(tmp_path):
    t = task_ids("retail", "test")
    f = write(tmp_path, "A-retail-test.jsonl", [rec(t[0]), rec(t[0], False, 1), rec(t[1], crashed=True)])
    e = pc.check(f)
    assert e["file"] == "conversations/A/A-retail-test.jsonl"
    counts = (e["conversations"], e["tasks"], e["max_samples_per_task"], e["crashed"], e["passed"])
    assert counts == (3, 2, 2, 1, 1)
    assert e["success"] == 0.5


def test_slices_of_train_count_as_train(tmp_path):
    f = write(tmp_path, "S2-retail-val.jsonl", [rec(task_ids("retail", "train")[0])])
    assert pc.check(f)["split"] == "val"


def test_a_train_file_may_hold_dev_tasks_and_counts_them(tmp_path):
    f = write(tmp_path, "A-telecom-train.jsonl", [rec(task_ids("telecom", "dev")[0])])
    assert pc.check(f)["dev_tasks_in_train_file"] == 1


def test_a_task_from_another_split_is_refused(tmp_path):
    f = write(tmp_path, "A-retail-dev.jsonl", [rec(task_ids("retail", "test")[0])])
    with pytest.raises(pc.CheckError, match="test task, not dev"):
        pc.check(f)


def test_a_line_with_a_key_is_refused(tmp_path):
    r = rec(task_ids("retail", "test")[0]) | {"note": "sk-" + "a" * 30}
    f = write(tmp_path, "A-retail-test.jsonl", [r])
    with pytest.raises(pc.CheckError, match="API key"):
        pc.check(f)


def test_dry_run_writes_the_dataset_and_uploads_nothing(tmp_path, out):
    src = tmp_path / "arms"
    src.mkdir()
    write(src, "A-retail-test.jsonl", [rec(task_ids("retail", "test")[0])])
    write(src, "S2E4-airline-test.jsonl", [rec(task_ids("airline", "test")[0], False)])
    write(src, "notes.jsonl", [{"x": 1}])  # not an arm's file: ignored
    assert pc.main(["--src", str(src), "--arms", "S2E4", "--dry-run"]) == 0
    assert (out / "conversations/S2E4/S2E4-airline-test.jsonl").exists()
    assert not (out / "conversations/A").exists()
    manifest = json.loads((out / "manifest.json").read_text())
    assert list(manifest) == ["conversations/S2E4/S2E4-airline-test.jsonl"]
    assert (out / "splits.json").exists()
    readme = (out / "README.md").read_text()
    assert "| S2E4 | SFT v2's data trained for 4 epochs (the final model) | test | 1 |" in readme


def test_one_bad_file_stops_everything(tmp_path, out):
    src = tmp_path / "arms"
    src.mkdir()
    write(src, "A-retail-test.jsonl", [rec(task_ids("retail", "test")[0])])
    write(src, "A-retail-dev.jsonl", [rec(task_ids("retail", "test")[0])])
    assert pc.main(["--src", str(src), "--dry-run"]) == 1
    assert not out.exists()
