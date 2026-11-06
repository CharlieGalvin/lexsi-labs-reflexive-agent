"""SEL-7: the hard subset groups every train task by Arm A's result and the teacher's fix."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import hard, splits
from reflexive.files import sha256
from reflexive.hallucination import domain_tools

bh = load_script("data/build_hard_subset")

Record = dict[str, Any]
CHAT = [
    {"role": "system", "content": "policy"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'd like to change my order, please."},
]


def arm(domain: str, task_id: str, passed: bool = False, **kw: Any) -> Record:
    return {
        "arm": "A",
        "domain": domain,
        "task_id": task_id,
        "sample": 0,
        "termination": "user_stop",
        "error": None,
        "student": "aya-expanse-32b",
        "outcome": {"passed": passed, "reward": float(passed), "checks": {"DB": float(passed)}},
        "agent_view": CHAT,
        **kw,
    }


def gold(domain: str, task_id: str) -> Record:
    return {"id": f"glm53f/{domain}/{task_id}/1", "domain": domain, "task_id": task_id, "split": "train"}


def train(domain: str) -> list[str]:
    return splits.task_ids(domain, "train")


SRC = {"retail": {"file": "A-retail-train.jsonl", "sha256_16": "0" * 16}}


def build_retail(**changes: Record) -> tuple[list[Record], dict[str, Any]]:
    """Every retail train task failed by Arm A, with a fix for all but the second; ``changes`` by task ID."""
    ids = train("retail")
    recs = [changes.get(t, arm("retail", t)) for t in ids]
    return hard.build(recs, [gold("retail", t) for t in ids if t != ids[1]], SRC)


def test_each_task_lands_in_its_group():
    ids = train("retail")
    recs, report = build_retail(**{ids[2]: arm("retail", ids[2], passed=True)})
    by = {r["task_id"]: r for r in recs}
    assert by[ids[0]]["group"] == "healable" and by[ids[0]]["golden_id"] == f"glm53f/retail/{ids[0]}/1"
    assert by[ids[1]]["group"] == "teacher_unsolved" and by[ids[1]]["golden_id"] is None
    assert by[ids[2]]["group"] == "solved_by_aya"
    assert report["total"]["groups"] == {"healable": len(ids) - 2, "teacher_unsolved": 1, "solved_by_aya": 1}


def test_records_follow_the_frozen_train_order_and_carry_arm_a_result():
    recs, _ = build_retail()
    assert [r["task_id"] for r in recs] == train("retail")
    r = recs[0]
    assert r["format"] == hard.FORMAT and r["split"] == "train" and r["id"] == f"armA/retail/{r['task_id']}"
    assert r["arm_a"]["passed"] == 0 and r["arm_a"]["samples"] == 1 and r["arm_a"]["source"] == SRC["retail"]
    assert r["messages"] == CHAT


def test_a_failure_with_no_structural_slip_is_labelled_none():
    recs, report = build_retail()
    assert recs[0]["labels"] == ["NONE"] and recs[0]["primary"] == "NONE"
    assert report["total"]["labels"]["healable"]["NONE"] == len(recs) - 1
    assert report["thin"] == ["TOOL", "ARG", "LOOP"]  # no healable example of any type here


def test_a_call_to_the_customers_phone_tool_is_a_tool_failure_marked_customer_tool():
    _, customer = domain_tools("telecom")
    t = train("telecom")[0]
    call = {"id": "c1", "type": "function", "function": {"name": customer[0], "arguments": "{}"}}
    view = [*CHAT, {"role": "assistant", "content": "", "tool_calls": [call]}]
    labels, detail = hard.labels_of(arm("telecom", t, agent_view=view))
    assert labels == ["TOOL"] and detail["customer_tool"] and detail["primary"] == "TOOL"


def test_dev_tasks_are_left_out_and_counted():
    dev = splits.task_ids("retail", "dev")[0]
    ids = train("retail")
    recs, report = hard.build(
        [*(arm("retail", t) for t in ids), arm("retail", dev)], [gold("retail", ids[0])], SRC
    )
    assert dev not in {r["task_id"] for r in recs} and report["dev_left_out"] == {"retail": 1}
    _, again = hard.build([*(arm("retail", t) for t in ids), arm("retail", dev), arm("retail", dev)], [], SRC)
    assert again["dev_left_out"] == {"retail": 1}  # tasks, not records


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (lambda: arm("retail", splits.task_ids("retail", "test")[0]), "TEST task"),
        (lambda: arm("retail", "no-such-task"), "in no split"),
        (lambda: arm("retail", train("retail")[0], termination="infrastructure_error"), "run broke"),
        (lambda: arm("retail", train("retail")[0], error="RateLimitError"), "run broke"),
        (lambda: arm("retail", train("retail")[0], outcome=None), "no saved outcome"),
        (lambda: arm("retail", train("retail")[0], outcome={"reward": 0.0}), "no saved outcome"),
        (lambda: arm("retail", train("retail")[1]), "more than one sample"),
    ],
)
def test_a_bad_arm_a_record_stops_the_build(bad, message):
    ids = train("retail")
    recs = [arm("retail", t) for t in ids[1:]] + [bad()]
    with pytest.raises(hard.HardSubsetError) as e:
        hard.build(recs, [], SRC)
    assert any(message in line for line in e.value.lines), e.value.lines


def test_a_train_task_without_an_arm_a_record_stops_the_build():
    recs = [arm("retail", t) for t in train("retail")[1:]]
    with pytest.raises(hard.HardSubsetError, match="1 train tasks without an Arm A record, 0 domains"):
        hard.build(recs, [], SRC)


def test_a_golden_record_that_isnt_train_or_is_doubled_stops_the_build():
    ids = train("retail")
    recs = [arm("retail", t) for t in ids]
    with pytest.raises(hard.HardSubsetError, match="isn't a train task"):
        hard.build(recs, [gold("retail", splits.task_ids("retail", "dev")[0])], SRC)
    with pytest.raises(hard.HardSubsetError, match="isn't a train task"):
        hard.build(recs, [gold("retail", "no-such-task")], SRC)
    with pytest.raises(hard.HardSubsetError, match="two golden fixes"):
        hard.build(recs, [gold("retail", ids[0]), {**gold("retail", ids[0]), "id": "other"}], SRC)


# --- just hard-subset ----------------------------------------------------------------------------


def _folders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> tuple[Path, Path, Path]:
    arms, golden, out = tmp_path / "arms", tmp_path / "golden", tmp_path / "out"
    arms.mkdir()
    golden.mkdir()
    fixes = []
    for d in splits.DOMAINS:
        ids = train(d)
        recs = [arm(d, t, passed=(i == 0)) for i, t in enumerate(ids)]
        if d == "telecom":
            recs.append(arm(d, splits.task_ids(d, "dev")[0]))  # telecom's file also holds dev tasks
        (arms / f"A-{d}-train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
        fixes += [gold(d, t) for t in ids[:-1]]
    (golden / "golden_train.jsonl").write_text("".join(json.dumps(g) + "\n" for g in fixes))
    report = {"files": {"golden_train.jsonl": sha256(golden / "golden_train.jsonl")}}
    (golden / "build_report.json").write_text(json.dumps(report))
    args = ["--arms", str(arms), "--golden", str(golden), "--out", str(out), *extra]
    monkeypatch.setattr(sys, "argv", ["build_hard_subset.py", *args])
    return arms, golden, out


def test_the_command_writes_the_subset_and_a_report_with_its_checksum(tmp_path, monkeypatch):
    _, _, out = _folders(tmp_path, monkeypatch)
    assert bh.main() == 0
    rows = [json.loads(x) for x in (out / bh.FILE).read_text().splitlines()]
    report = json.loads((out / bh.REPORT).read_text())
    assert len(rows) == sum(len(train(d)) for d in splits.DOMAINS)
    assert report["files"][bh.FILE] == sha256(out / bh.FILE)
    assert report["dev_left_out"]["telecom"] == 1
    assert (
        report["total"]["groups"]["solved_by_aya"] == 3 and report["total"]["groups"]["teacher_unsolved"] == 3
    )


def test_the_command_refuses_to_replace_a_build_without_force(tmp_path, monkeypatch, capsys):
    _folders(tmp_path, monkeypatch)
    assert bh.main() == 0
    assert bh.main() == 1 and "--force" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--force"])
    assert bh.main() == 0


def test_the_command_refuses_golden_files_changed_since_their_build(tmp_path, monkeypatch, capsys):
    _, golden, out = _folders(tmp_path, monkeypatch)
    with (golden / "golden_train.jsonl").open("a") as f:
        f.write("\n")
    assert bh.main() == 1
    assert "changed since `just golden`" in capsys.readouterr().out and not out.exists()


def test_the_command_refuses_a_file_of_another_arm(tmp_path, monkeypatch, capsys):
    arms, _, _ = _folders(tmp_path, monkeypatch)
    path = arms / "A-airline-train.jsonl"
    path.write_text(path.read_text().replace('"arm": "A"', '"arm": "C"', 1))
    assert bh.main() == 1 and "other arms or domains" in capsys.readouterr().out


def test_the_command_reports_a_missing_golden_build_report(tmp_path, monkeypatch, capsys):
    _, golden, _ = _folders(tmp_path, monkeypatch)
    (golden / "build_report.json").unlink()
    assert bh.main() == 1 and "can't read" in capsys.readouterr().out
