"""SEL-21: single-turn GRPO prompts from the training conversations (reagent.sft/v1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import grpo_data, splits

bg = load_script("data/build_grpo_data")

Record = dict[str, Any]
ACTION = 'Action: ```json\n[\n    {"tool_name": "get_user_details", "parameters": {"user_id": "u1"}}\n]\n```'
TWO = (
    'Action: ```json\n[{"tool_name": "get_order_details", "parameters": {"order_id": "#W1"}},'
    ' {"tool_name": "get_order_details", "parameters": {"order_id": "#W2"}}]\n```'
)
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
        {"role": "assistant", "content": TWO},
        {"role": "user", "content": "<results>\n...\n</results>"},
        {"role": "assistant", "content": "Done: your order is changed."},
        {"role": "user", "content": "Thanks! ###STOP###"},
    ]


def conv(task: tuple[str, str], group: str = "healable", **kw: Any) -> Record:
    msgs = chat()
    r = {
        "id": f"glm53f/{task[0]}/{task[1]}/1",
        "format": "reagent.sft/v1",
        "domain": task[0],
        "task_id": task[1],
        "group": group,
        "labels": ["ARG"],
        "messages": msgs,
        "train_on": [i for i, m in enumerate(msgs) if m["role"] == "assistant"],
    }
    r.update(kw)
    return r


def test_one_prompt_per_tool_call_turn_by_default():
    rows = grpo_data.turns_of(conv(TRAIN[0]))
    assert [r["turn"] for r in rows] == [3, 5]
    r = rows[0]
    assert r["format"] == grpo_data.FORMAT and r["kind"] == "call"
    assert r["prompt"] == chat()[:3] and r["prompt"][-1]["role"] == "user"
    assert r["target"] == ACTION
    assert r["target_calls"] == [{"name": "get_user_details", "arguments": {"user_id": "u1"}}]
    assert r["id"] == f"glm53f/retail/{TRAIN[0][1]}/1#t3"
    assert len(rows[1]["target_calls"]) == 2


def test_text_turns_only_when_asked():
    rows = grpo_data.turns_of(conv(TRAIN[0]), kinds=("call", "text"))
    assert [(r["turn"], r["kind"]) for r in rows] == [(1, "text"), (3, "call"), (5, "call"), (7, "text")]
    assert rows[0]["prompt"] == chat()[:1] and rows[0]["target_calls"] == []


def test_a_single_target_heal_example_gives_one_prompt():
    """`heal.to_sft_example` writes the fixed turn as the only target."""
    msgs = chat()[:4]
    rows = grpo_data.turns_of(conv(TRAIN[0], messages=msgs, train_on=[3]))
    assert [r["turn"] for r in rows] == [3]


def test_build_reports_and_keeps_the_sets_apart():
    train, heldout, report = grpo_data.build(
        [conv(TRAIN[0]), conv(TRAIN[1])], [conv(TRAIN[2], "solved_by_aya")]
    )
    assert len(train) == 4 and len(heldout) == 2
    assert report["train"]["conversations"] == 2 and report["train"]["tasks"] == 2
    assert report["train"]["kinds"] == {"call": 4} and report["train"]["multi_call_turns"] == 2
    assert report["train"]["tools"] == {"get_order_details": 4, "get_user_details": 2}
    assert report["heldout"]["prompts"] == 2


def test_a_task_in_both_sets_is_refused():
    with pytest.raises(grpo_data.GrpoDataError, match="both train and held out"):
        grpo_data.build([conv(TRAIN[0])], [conv(TRAIN[0], "solved_by_aya", id="other")])


def test_the_same_conversation_twice_is_refused():
    with pytest.raises(grpo_data.GrpoDataError, match="given twice"):
        grpo_data.build([conv(TRAIN[0]), conv(TRAIN[0])], [conv(TRAIN[1])])


@pytest.mark.parametrize("split", ["dev", "test"])
def test_only_train_tasks(split: str):
    task = ("retail", splits.task_ids("retail", split)[0])
    with pytest.raises(grpo_data.GrpoDataError) as e:
        grpo_data.build([conv(task)], [])
    assert f"is a {split} task" in "\n".join(e.value.lines)


@pytest.mark.parametrize(
    ("change", "says"),
    [
        ({"format": "reagent.golden/v1"}, "expected 'reagent.sft/v1'"),
        ({"train_on": [2]}, "isn't an agent turn"),
        ({"train_on": [99]}, "outside the conversation"),
        ({"train_on": []}, "train_on is empty"),
        ({"messages": chat()[1:], "train_on": [0]}, "system prompt"),
        ({"task_id": "no-such-task"}, "unknown task"),
    ],
)
def test_bad_conversations_are_refused(change: Record, says: str):
    with pytest.raises(grpo_data.GrpoDataError) as e:
        grpo_data.build([conv(TRAIN[0], **change)], [])
    assert says in "\n".join(e.value.lines)


def test_a_prompt_ending_on_an_agent_turn_is_refused():
    msgs = [*chat()[:4], {"role": "assistant", "content": ACTION}]
    with pytest.raises(grpo_data.GrpoDataError) as e:
        grpo_data.build([conv(TRAIN[0], messages=msgs, train_on=[4])], [])
    assert "ends on an agent turn" in "\n".join(e.value.lines)


def test_unknown_kinds_are_refused():
    with pytest.raises(grpo_data.GrpoDataError, match="kinds"):
        grpo_data.build([conv(TRAIN[0])], [], kinds=("calls",))


def write(path: Path, rows: list[Record]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_script_writes_both_files_and_the_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    healable = [conv(TRAIN[0]), conv(TRAIN[1])]
    train = write(tmp_path / "sft-healable.jsonl", healable)
    every = write(tmp_path / "sft-all.jsonl", [*healable, conv(TRAIN[2], "solved_by_aya")])
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["x", "--train", str(train), "--all", str(every), "--out", str(out)])
    assert bg.main() == 0
    rows = [json.loads(line) for line in (out / "grpo-healable.jsonl").read_text().splitlines()]
    held = [json.loads(line) for line in (out / "grpo-healable.heldout.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and len(held) == 2 and {r["task_id"] for r in held} == {TRAIN[2][1]}
    report = json.loads((out / "grpo-healable.report.json").read_text())
    assert report["format"] == grpo_data.FORMAT and report["inputs"]["heldout"]["group"] == "solved_by_aya"
    assert set(report["files"]) == {"grpo-healable.jsonl", "grpo-healable.heldout.jsonl"}
    assert "4 prompts from 2 conversations" in capsys.readouterr().out
    assert bg.main() == 1 and "--force" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--force"])
    assert bg.main() == 0


def test_script_says_to_build_the_training_data_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.setattr(sys, "argv", ["x", "--train", str(tmp_path / "nope.jsonl"), "--out", str(tmp_path)])
    assert bg.main() == 1 and "just train-data" in capsys.readouterr().out
