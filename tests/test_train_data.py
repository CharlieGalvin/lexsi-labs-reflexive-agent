"""SEL-66: training examples from golden v1.1 (the healable fixes, as Aya reads them)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import golden, splits, train_data
from reflexive.files import sha256
from reflexive.teacher import parse_prompted

bt = load_script("data/build_train_data")

Record = dict[str, Any]
ACTION = 'Action: ```json\n[\n    {"tool_name": "get_user_details", "parameters": {"user_id": "u1"}}\n]\n```'
TRAIN = [("retail", t) for t in splits.task_ids("retail", "train")[:4]]


def chat() -> list[Record]:
    return [
        {"role": "system", "content": "policy and tools"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Change my order, please. I'm u1."},
        {"role": "assistant", "content": ACTION},
        {
            "role": "user",
            "content": "<results>\nDocument: 0\ntool_name: get_user_details\nresult: {}\n</results>",
        },
        {"role": "assistant", "content": "Done: your order is changed."},
        {"role": "user", "content": "Thanks! ###STOP###"},
    ]


def gold(task: tuple[str, str], **kw: Any) -> Record:
    msgs = chat()
    r = {
        "id": f"glm53f/{task[0]}/{task[1]}/1",
        "format": golden.FORMAT,
        "domain": task[0],
        "task_id": task[1],
        "split": "train",
        "teacher": "z-ai/glm-5.3-flash",
        "messages": msgs,
        "train_on": [i for i, m in enumerate(msgs) if m["role"] == "assistant"],
    }
    r.update(kw)
    return r


def hard(task: tuple[str, str], group: str, fix: bool = True, **kw: Any) -> Record:
    r = {
        "id": f"armA/{task[0]}/{task[1]}",
        "domain": task[0],
        "task_id": task[1],
        "split": "train",
        "group": group,
        "golden_id": f"glm53f/{task[0]}/{task[1]}/1" if fix else None,
        "labels": ["ARG"] if group != "solved_by_aya" else ["NONE"],
    }
    r.update(kw)
    return r


def inputs() -> tuple[list[Record], list[Record]]:
    """4 train tasks: 2 healable, 1 teacher-unsolved (no fix), 1 that Aya already solves (with a fix)."""
    hs = [
        hard(TRAIN[0], "healable"),
        hard(TRAIN[1], "teacher_unsolved", fix=False),
        hard(TRAIN[2], "healable"),
        hard(TRAIN[3], "solved_by_aya"),
    ]
    return [gold(t) for t in (TRAIN[0], TRAIN[2], TRAIN[3])], hs


def test_the_action_fixture_parses_like_a_real_golden_turn():
    assert parse_prompted(ACTION)


def test_healable_picks_only_the_healable_fixes_in_frozen_order():
    gs, hs = inputs()
    examples, report = train_data.build(gs, hs)
    assert [e["task_id"] for e in examples] == [TRAIN[0][1], TRAIN[2][1]]
    assert report["groups"] == {"healable": 2} and report["labels"]["ARG"] == 2
    e = examples[0]
    assert e["format"] == train_data.FORMAT and e["messages"] == chat() and e["train_on"] == [1, 3, 5]
    assert e["source"] == {
        "golden_id": e["id"],
        "hard_id": f"armA/retail/{TRAIN[0][1]}",
        "teacher": "z-ai/glm-5.3-flash",
    }
    assert report["agent_turns"] == 6 and report["approx_tokens"]["max"] > 0


def test_all_adds_the_fixes_for_tasks_aya_already_solves():
    gs, hs = inputs()
    examples, report = train_data.build(gs, hs, select="all")
    assert [e["group"] for e in examples] == ["healable", "healable", "solved_by_aya"]


def test_an_unknown_selection_is_refused():
    with pytest.raises(ValueError, match="select must be"):
        train_data.build(*inputs(), select="everything")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda g: g.update(train_on=[1, 3]), "train_on isn't exactly the agent's turns"),
        (lambda g: g.update(train_on=[1, 2, 3, 5]), "train_on isn't exactly the agent's turns"),
        (lambda g: g["messages"].pop(0), "doesn't start with the system prompt"),
        (lambda g: g["messages"].insert(2, {"role": "user", "content": "again"}), "two turns in a row"),
        (lambda g: g["messages"].insert(2, {"role": "tool", "content": "{}"}), "isn't assistant or user"),
        (lambda g: g["messages"][3].update(content="Action: ```json\n[{broken\n```"), "doesn't parse"),
        (lambda g: g["messages"][5].update(content="  "), "agent turn 5 is empty"),
        (lambda g: g.update(format="reagent.golden/v0"), "expected reagent.golden/v1"),
        (lambda g: g.update(split="dev"), "only train tasks are trained on"),
    ],
)
def test_a_golden_fix_that_isnt_clean_stops_the_build(edit, message):
    gs, hs = inputs()
    edit(gs[0])
    with pytest.raises(train_data.TrainDataError) as e:
        train_data.build(gs, hs)
    assert any(message in line for line in e.value.lines), e.value.lines


@pytest.mark.parametrize("split", ["dev", "test"])
def test_a_dev_or_test_task_in_the_hard_subset_stops_the_build(split):
    gs, hs = inputs()
    other = (TRAIN[0][0], splits.task_ids("retail", split)[0])
    hs.append(hard(other, "healable", split=split))
    gs.append(gold(other, split=split))
    with pytest.raises(train_data.TrainDataError) as e:
        train_data.build(gs, hs)
    assert any(f"a {split} task, not train" in line for line in e.value.lines)


def test_a_fix_that_is_missing_or_for_another_task_stops_the_build():
    gs, hs = inputs()
    with pytest.raises(train_data.TrainDataError) as e:
        train_data.build(gs[1:], hs)
    assert any("isn't in golden_train.jsonl" in line for line in e.value.lines)
    gs[0]["task_id"] = TRAIN[1][1]
    with pytest.raises(train_data.TrainDataError) as e:
        train_data.build(gs, hs)
    assert any("the hard subset says" in line for line in e.value.lines)


def test_a_fix_listed_twice_or_unaccounted_for_stops_the_build():
    gs, hs = inputs()
    with pytest.raises(train_data.TrainDataError, match="appears twice"):
        train_data.build([*gs, gs[0]], hs)
    with pytest.raises(train_data.TrainDataError) as e:
        train_data.build(gs, hs[:3], select="all")  # the solved task's fix has no hard-subset record
    assert any("the hard subset doesn't list" in line for line in e.value.lines)


def test_nothing_selected_stops_the_build():
    gs, hs = inputs()
    with pytest.raises(train_data.TrainDataError, match="no fixes selected"):
        train_data.build(gs, [h for h in hs if h["group"] != "healable"])


# --- just train-data -----------------------------------------------------------------------------


def _write(folder: Path, name: str, rows: list[Record]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder / name


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *args: str, report: bool = True) -> Path:
    gs, hs = inputs()
    g = _write(tmp_path / "v1", "golden_train.jsonl", gs)
    h = _write(tmp_path / "hard", "hard_subset.jsonl", hs)
    if report:
        (h.parent / "hard_report.json").write_text(json.dumps({"inputs": {"golden": {"sha256": sha256(g)}}}))
    out = tmp_path / "train"
    argv = ["build_train_data.py", "--golden", str(g), "--hard", str(h), "--out", str(out), *args]
    monkeypatch.setattr(sys, "argv", argv)
    return out


def test_the_command_writes_the_examples_and_a_report_with_checksums(tmp_path, monkeypatch):
    out = _setup(tmp_path, monkeypatch)
    assert bt.main() == 0
    rows = [json.loads(x) for x in (out / "sft-healable.jsonl").read_text().splitlines()]
    report = json.loads((out / "sft-healable.report.json").read_text())
    assert len(rows) == 2 and report["examples"] == 2 and report["select"] == "healable"
    assert report["files"]["sft-healable.jsonl"] == sha256(out / "sft-healable.jsonl")
    assert report["inputs"]["golden"]["sha256"] == sha256(tmp_path / "v1" / "golden_train.jsonl")


def test_the_command_refuses_to_replace_a_build_without_force(tmp_path, monkeypatch, capsys):
    _setup(tmp_path, monkeypatch)
    assert bt.main() == 0
    assert bt.main() == 1 and "--force" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--force"])
    assert bt.main() == 0


def test_the_command_reads_a_dataset_snapshot_folder(tmp_path, monkeypatch):
    gs, hs = inputs()
    snap = tmp_path / "hub-v1.1"
    _write(snap, "golden_train.jsonl", gs)
    _write(snap, "hard_subset.jsonl", hs)  # no hard_report.json in the HF snapshot
    out = tmp_path / "train"
    monkeypatch.setattr(sys, "argv", ["x", "--dataset", str(snap), "--out", str(out), "--select", "all"])
    assert bt.main() == 0 and len((out / "sft-all.jsonl").read_text().splitlines()) == 3


def test_the_command_refuses_a_hard_subset_built_from_other_golden_files(tmp_path, monkeypatch, capsys):
    out = _setup(tmp_path, monkeypatch)
    (tmp_path / "hard" / "hard_report.json").write_text(json.dumps({"inputs": {"golden": {"sha256": "old"}}}))
    assert bt.main() == 1 and "built from another golden_train.jsonl" in capsys.readouterr().out
    assert not out.exists()


def test_the_command_reports_missing_inputs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["x", "--dataset", str(tmp_path / "nowhere"), "--out", str(tmp_path / "o")]
    )
    assert bt.main() == 1 and "doesn't exist" in capsys.readouterr().out
