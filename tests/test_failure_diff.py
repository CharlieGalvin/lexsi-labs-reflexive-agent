"""The comparison behind `just arm-failures` (no τ²-bench needed)."""

import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "failure_diff", Path(__file__).parents[1] / "scripts/arms/failure_diff.py"
)
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)


def test_compare_sorts_ok_wrong_missing_and_extra():
    want = [
        ("return", {"order": "#1", "items": ["a"]}),
        ("exchange", {"order": "#2"}),
        ("cancel", {"order": "#3"}),
    ]
    did = [
        ("return", {"order": "#1", "items": ["a"]}),
        ("exchange", {"order": "#9"}),
        ("modify", {"order": "#4"}),
    ]
    c = fd.compare(want, did)
    assert c["ok"] == [want[0]]
    assert c["wrong"] == [(want[1], did[1])]
    assert c["missing"] == [want[2]]
    assert c["extra"] == [did[2]]


def test_exact_matches_win_over_same_tool_pairs():
    want = [("exchange", {"order": "#1"}), ("exchange", {"order": "#2"})]
    did = [("exchange", {"order": "#2"}), ("exchange", {"order": "#7"})]
    c = fd.compare(want, did)
    assert c["ok"] == [("exchange", {"order": "#2"})]
    assert c["wrong"] == [(("exchange", {"order": "#1"}), ("exchange", {"order": "#7"}))]


def test_args_diff_names_only_what_differs():
    assert fd.args_diff({"a": 1, "b": 2}, {"a": 1, "b": 3}) == "b: 2 → 3"


def test_agent_calls_reads_json_arguments_and_skips_the_customer():
    msgs = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "f", "arguments": '{"x": 1}'}}]},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "g", "arguments": "not json"}}]},
    ]
    assert fd.agent_calls(msgs) == [("f", {"x": 1}), ("g", {"<unparsed>": "not json"})]


def test_latest_keeps_the_rerun_of_a_crashed_conversation(tmp_path):
    f = tmp_path / "S-retail-dev.jsonl"
    rows = [
        {"task_id": "1", "outcome": {"passed": False}, "error": "Connection error"},
        {"task_id": "2", "outcome": {"passed": True}},
        {"task_id": "1", "outcome": {"passed": True}},
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = {r["task_id"]: r["outcome"]["passed"] for r in fd.latest(f)}
    assert got == {"1": True, "2": True}


def test_mentioned_recognises_plain_words_for_phone_actions():
    text = "please turn off airplane mode, then restart your phone"
    assert fd.mentioned("toggle_airplane_mode", text)
    assert fd.mentioned("reboot_device", text)
    assert not fd.mentioned("reset_apn_settings", text)
