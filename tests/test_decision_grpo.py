"""Decision GRPO (arm H2): pending expected changes and the decision reward."""

from __future__ import annotations

import json

from reflexive import decision_data as dd
from reflexive import heal_rewards
from reflexive.grpo_data import FORMAT

WRITES = ["cancel_pending_order", "modify_pending_order_items"]
CANCEL = ["cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}, None]
MODIFY = ["modify_pending_order_items", {"order_id": "#W2", "item_ids": ["1"], "new_item_ids": ["2"]}, None]


def call(name: str, **args: object) -> dict:
    return {"name": name, "arguments": args}


def action(name: str, **args: object) -> str:
    return "Action: ```json\n" + json.dumps([{"tool_name": name, "parameters": args}]) + "\n```"


def result(name: str, text: str) -> str:
    return f"<results>\nDocument: 0\ntool_name: {name}\nresult: {text}\n</results>"


COLUMN = {"pending": [CANCEL, MODIFY], "writes": WRITES}


def test_a_change_nobody_asked_for_scores_zero():
    s, found = dd.decision_score([call("modify_pending_order_items", order_id="#W9")], [], COLUMN)
    assert s == 0.0 and found == ["modify_pending_order_items: wrong arguments"]
    s, found = dd.decision_score(
        [call("cancel_pending_order", order_id="#W1", reason="ordered by mistake")],
        [],
        {"pending": [MODIFY], "writes": WRITES},
    )
    assert s == 0.0 and found == ["cancel_pending_order: not expected"]


def test_making_the_change_where_the_teacher_did():
    teacher = [call("cancel_pending_order", order_id="#W1", reason="no longer needed")]
    assert dd.decision_score(teacher, teacher, COLUMN) == (1.0, ["made"])
    assert dd.decision_score([], teacher, COLUMN) == (dd.HOLD_OFF, ["held off"])  # asked instead


def test_no_change_where_none_was_made_and_reads_are_free():
    assert dd.decision_score([call("get_order_details", order_id="#W1")], [], COLUMN) == (1.0, [])
    early = [call("cancel_pending_order", order_id="#W1", reason="no longer needed")]
    assert dd.decision_score(early, [], COLUMN) == (dd.EARLY, ["early"])


def test_no_column_is_neutral_so_h_is_unchanged():
    assert dd.decision_score([call("cancel_pending_order", order_id="#X")], [], None) == (1.0, [])


def test_pending_drops_changes_already_made_but_not_failed_ones():
    prompt = [
        {"role": "system", "content": "s"},
        {
            "role": "assistant",
            "content": action("cancel_pending_order", order_id="#W1", reason="no longer needed"),
        },
        {"role": "user", "content": result("cancel_pending_order", '{"status": "cancelled"}')},
        {
            "role": "assistant",
            "content": action(
                "modify_pending_order_items", order_id="#W2", item_ids=["1"], new_item_ids=["2"]
            ),
        },
        {"role": "user", "content": result("modify_pending_order_items", "Error: payment method not found")},
    ]
    assert dd.pending([CANCEL, MODIFY], prompt) == [MODIFY]


def test_compare_args_limit_what_must_match():
    e = ["cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}, ["order_id"]]
    assert dd.same_call(e, "cancel_pending_order", {"order_id": "#W1", "reason": "ordered by mistake"})
    assert not dd.same_call(e, "cancel_pending_order", {"order_id": "#W2", "reason": "no longer needed"})


def test_heal_rewards_score_the_decision_from_the_row():
    row = {
        "target": action("cancel_pending_order", order_id="#W1", reason="no longer needed"),
        "target_calls": json.dumps([call("cancel_pending_order", order_id="#W1", reason="no longer needed")]),
        "domain": "retail",
        "prompt": [{"role": "system", "content": "s"}],
        "decision": json.dumps(COLUMN),
    }
    wrong = heal_rewards.score_row(
        action("cancel_pending_order", order_id="#W7", reason="no longer needed"), row
    )
    right = heal_rewards.score_row(row["target"], row)
    assert wrong["decision"] == 0.0 and right["decision"] == 1.0
    assert "decision" in heal_rewards.COMPONENTS and heal_rewards.DEFAULT_WEIGHTS["decision"] == 0.0


def test_the_trainer_keeps_the_decision_column(tmp_path, monkeypatch):
    import importlib.util
    import sys

    from reflexive.files import ROOT

    spec = importlib.util.spec_from_file_location(
        "ts_decision", ROOT / "scripts" / "grpo" / "train_single_turn.py"
    )
    ts = importlib.util.module_from_spec(spec)
    sys.modules["ts_decision"] = ts
    spec.loader.exec_module(ts)
    row = {
        "prompt": [{"role": "system", "content": "s"}],
        "target": "ok",
        "target_calls": [],
        "domain": "retail",
        "id": "x#t1",
        "format": FORMAT,
        "task_id": __import__("reflexive.splits", fromlist=["task_ids"]).task_ids("retail", "train")[0],
        "failure_type": "DECISION",
        "failed": {"calls": [], "text": ""},
        "history": {"calls": [], "texts": []},
        "decision": COLUMN,
    }
    data = tmp_path / "d.jsonl"
    data.write_text(json.dumps(row) + "\n")
    out = ts.prompts({"data": str(data), "rewards": "heal", "reward_weights": {"decision": 2.0}})
    assert json.loads(out[0]["decision"]) == COLUMN
