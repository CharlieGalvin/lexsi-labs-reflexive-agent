"""SEL-21: the single-turn GRPO reward (one sampled agent turn against the teacher's)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import grpo_rewards as g

check = load_script("data/check_grpo_rewards")

Row = dict[str, Any]
PROMPT = [
    {"role": "system", "content": "policy and tools"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "I'm Yusuf Rossi, zip 19122. Where is my order #W2378156?"},
]
FIND: Row = {
    "name": "find_user_id_by_name_zip",
    "arguments": {"first_name": "Yusuf", "last_name": "Rossi", "zip": "19122"},
}
ORDER: Row = {"name": "get_order_details", "arguments": {"order_id": "#W2378156"}}
BEST = sum(g.DEFAULT_WEIGHTS.values())


def act(*calls: Row) -> str:
    return check.action(list(calls))


def s(text: str, target: list[Row] | None = None, domain: str = "retail", teacher: str = "") -> Row:
    target = [FIND] if target is None else target
    return g.score(text, target, domain, PROMPT, teacher or (act(*target) if target else "Sure, one moment."))


def test_the_teachers_turn_scores_the_maximum():
    r = s(act(FIND))
    assert all(r[k] == 1.0 for k in g.COMPONENTS) and g.total(r, g.DEFAULT_WEIGHTS) == pytest.approx(BEST)


def test_case_spaces_and_number_formats_dont_count():
    sample = {"name": FIND["name"], "arguments": {"first_name": " yusuf", "last_name": "ROSSI", "zip": 19122}}
    assert s(act(sample))["args_match"] == 1.0


def test_a_wrong_argument_loses_its_share():
    sample = {"name": FIND["name"], "arguments": {**FIND["arguments"], "zip": "19123"}}
    r = s(act(sample))
    assert r["tool_match"] == 1.0 and r["args_match"] == pytest.approx(2 / 3)


def test_an_invented_id_is_ungrounded():
    r = s(act({"name": "get_order_details", "arguments": {"order_id": "#W9999999"}}), [ORDER])
    assert r["grounded"] == 0.0 and "ungrounded" in r["findings"] and r["args_match"] == 0.0


def test_a_placeholder_is_ungrounded():
    r = s(act({"name": "get_order_details", "arguments": {"order_id": "<order_id>"}}), [ORDER])
    assert r["grounded"] == 0.0 and "placeholder" in r["findings"]


def test_a_missing_parameter_is_invalid():
    r = s(act({"name": FIND["name"], "arguments": {"first_name": "Yusuf", "last_name": "Rossi"}}))
    assert r["valid_call"] == 0.0 and "missing_param" in r["findings"]


def test_an_unknown_tool_is_invalid():
    r = s(act({"name": "lookup_customer", "arguments": {}}))
    assert r["valid_call"] == 0.0 and r["tool_match"] == 0.0 and r["unknown_tools"] == ["lookup_customer"]


def test_a_customer_only_tool_is_invalid_and_named():
    target = [{"name": "get_customer_by_phone", "arguments": {"phone_number": "19122"}}]
    r = s(act({"name": "toggle_airplane_mode", "arguments": {}}), target, domain="telecom")
    assert r["valid_call"] == 0.0 and r["customer_tools"] == ["toggle_airplane_mode"]


def test_the_call_inside_prose_loses_only_format():
    r = s(f"Sure, let me look that up. {act(FIND)}\nTool result: user_1")
    assert r["format"] == 0.0 and r["tool_match"] == r["args_match"] == r["valid_call"] == 1.0


def test_tool_overlap_gives_partial_credit_and_pairs_by_arguments():
    other = {"name": "get_order_details", "arguments": {"order_id": "#W1"}}
    r = s(act(ORDER), [other, ORDER])
    assert r["tool_match"] == pytest.approx(0.5) and r["args_match"] == pytest.approx(0.5)
    swapped = s(act(ORDER, FIND), [FIND, ORDER])  # order within a turn doesn't matter
    assert swapped["tool_match"] == swapped["args_match"] == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "OK.",
        "Thanks, goodbye! ###STOP###",
        act({"name": "transfer_to_human_agents", "arguments": {"summary": "needs help"}}),
    ],
)
def test_giving_up_is_penalised(text: str):
    assert s(text)["no_give_up"] == 0.0


def test_a_hand_off_the_teacher_also_made_is_not_giving_up():
    handoff = {"name": "transfer_to_human_agents", "arguments": {"summary": "Yusuf Rossi wants a refund"}}
    r = s(act(handoff), [handoff])
    assert r["no_give_up"] == 1.0 and g.total(r, g.DEFAULT_WEIGHTS) == pytest.approx(BEST)


def test_asking_the_customer_is_valid_but_not_the_action():
    r = s("Could you please confirm the order number you'd like me to check?")
    assert r["valid_call"] == r["no_give_up"] == 1.0 and r["tool_match"] == r["args_match"] == 0.0


def test_a_text_reference_rewards_text_and_penalises_a_call():
    talk = s("I've found your order: it shipped yesterday.", target=[])
    call = s(act(ORDER), target=[])
    assert talk["tool_match"] == talk["args_match"] == 1.0
    assert call["tool_match"] == call["args_match"] == 0.0


def test_unreadable_arguments_count_as_none():
    text = 'Action: ```json\n[{"tool_name": "get_order_details", "parameters": "#W2378156"}]\n```'
    r = s(text, [ORDER])
    assert r["args_match"] < 1.0


@pytest.mark.parametrize(
    "completion",
    [act(FIND), [{"role": "assistant", "content": act(FIND)}], {"role": "assistant", "content": act(FIND)}],
)
def test_completions_as_text_or_messages(completion: Any):
    assert g.completion_text(completion) == act(FIND)


def test_the_reward_fn_reads_trl_columns_and_logs(tmp_path: Path):
    log = tmp_path / "rewards.jsonl"
    fn = g.make_reward_fn({"format": 0.0}, log_path=log)
    out = fn(
        completions=[act(FIND), "OK."],
        prompts=[PROMPT, PROMPT],
        target_calls=[json.dumps([FIND])] * 2,
        target=[act(FIND)] * 2,
        domain=["retail"] * 2,
        id=["a#t3", "a#t3"],
    )
    assert out[0] == pytest.approx(BEST - g.DEFAULT_WEIGHTS["format"]) and out[1] < out[0]
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["id"] for r in rows] == ["a#t3", "a#t3"] and rows[1]["no_give_up"] == 0.0


def test_each_component_is_logged_as_a_metric_when_trl_asks():
    seen: dict[str, float] = {}
    g.make_reward_fn()(
        completions=[act(FIND), "OK."],
        prompts=[PROMPT, PROMPT],
        target_calls=[[FIND]] * 2,
        domain=["retail"] * 2,
        log_metric=seen.__setitem__,
    )
    assert set(seen) == {f"reward_parts/{k}" for k in g.COMPONENTS}
    assert seen["reward_parts/tool_match"] == pytest.approx(0.5)


def test_a_previous_runs_log_is_kept_apart(tmp_path: Path):
    log = tmp_path / "rewards.jsonl"
    log.write_text('{"old": 1}\n')
    g.make_reward_fn(log_path=log)
    assert not log.exists() and len(list(tmp_path.glob("rewards.stopped-*.jsonl"))) == 1


def test_unknown_weights_and_missing_columns_are_refused():
    with pytest.raises(ValueError, match="unknown reward component"):
        g.make_reward_fn({"correct": 1.0})
    with pytest.raises(ValueError, match="target_calls"):
        g.make_reward_fn()(completions=["x"])


def test_check_script_passes_on_teacher_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    row = {"id": "x#t3", "domain": "retail", "prompt": PROMPT, "target": act(FIND), "target_calls": [FIND]}
    data = tmp_path / "grpo.jsonl"
    data.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", str(data)])
    assert check.main() == 0
    out = capsys.readouterr().out
    assert "1/1 prompts score the maximum" in out and "wrong tool" in out


def test_check_script_flags_a_reference_that_isnt_the_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    bad = {"name": "get_order_details", "arguments": {"order_id": "#W0000000"}}  # never in the prompt
    row = {"id": "x#t3", "domain": "retail", "prompt": PROMPT, "target": act(bad), "target_calls": [bad]}
    data = tmp_path / "grpo.jsonl"
    data.write_text(json.dumps(row) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", str(data)])
    assert check.main() == 1 and "⛔ x#t3" in capsys.readouterr().out
