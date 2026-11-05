"""reflexive.metrics and scripts/arms/metrics.py: one run's summary, from its records."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import splits
from reflexive.metrics import arm_metrics

Record = dict[str, Any]


def rec(task: str, passed: bool, *, domain: str = "retail", **kw: Any) -> Record:
    r: Record = {
        "domain": domain,
        "task_id": task,
        "arm": "A",
        "agent_llm": "c4ai-aya-expanse-32b",
        "user_llm": "gpt-4.1",
        "prompt_variant": "default",
        "prompt_version": "tau2/default",
        "termination": "user_stop",
        "outcome": {"passed": passed, "judged": True, "skipped": []},
        "flags": {"context_overflow": False},
        "diagnostics": {"tools_ran": 4, "unseen_arguments": [], "unknown_tools": 0, "repeated_failures": 0},
        "agent_usage": {
            "calls": 10,
            "prompt_tokens": 900,
            "completion_tokens": 100,
            "est_cost": 0.04,
            "cut_off": 0,
        },
        "user_cost": 0.01,
        "messages": [],
        "error": None,
    }
    for k, v in kw.items():
        r[k] = {**r[k], **v} if isinstance(v, dict) and isinstance(r.get(k), dict) else v
    return r


def test_a_crashed_conversation_is_counted_apart_and_not_scored() -> None:
    recs = [rec("1", True), rec("2", False), rec("3", False, error="RateLimitError: 429")]
    m = arm_metrics(recs)
    assert (m["conversations"], m["crashed"], m["passed"]) == (2, 1, 1)
    assert m["task_success"] == 0.5
    assert m["student_calls"] == 30  # a crashed conversation's calls were still made (and paid for)


def test_the_flags_count_conversations_not_events() -> None:
    recs = [
        rec("1", False, diagnostics={"unknown_tools": 3, "repeated_failures": 2}),
        rec("2", False, diagnostics={"unseen_arguments": ["#W0000000", "sara_doe_496"]}),
        rec("3", True),
    ]
    m = arm_metrics(recs)
    assert m["conversations_flagged"] == {
        "tool_unknown_tool": 1,
        "arg_unseen_id_like_argument": 1,
        "loop_repeated_failed_call": 1,
        "tool_or_arg": 2,
    }
    assert m["hallucination_rate_tool_or_arg"] == round(2 / 3, 4)


def test_a_hand_off_is_counted_from_the_agents_tool_call() -> None:
    call = {"role": "assistant", "tool_calls": [{"function": {"name": "transfer_to_human_agents"}}]}
    customer_said = {"role": "user", "content": "transfer_to_human_agents"}  # words aren't a call
    m = arm_metrics([rec("1", False, messages=[call]), rec("2", False, messages=[customer_said])])
    assert m["transferred_to_human"] == 1


def test_telecom_passes_are_broken_down_by_issue() -> None:
    recs = [
        rec("[mms_issue]a", True, domain="telecom"),
        rec("[mms_issue]b", False, domain="telecom"),
        rec("[service_issue]c", False, domain="telecom"),
    ]
    assert arm_metrics(recs)["by_issue"] == {
        "mms_issue": {"passed": 1, "n": 2},
        "service_issue": {"passed": 0, "n": 1},
    }
    assert arm_metrics([rec("1", True)])["by_issue"] is None  # retail has no issue types


def test_cost_tokens_and_calls_per_success() -> None:
    m = arm_metrics([rec("1", True), rec("2", False)])
    assert m["cost_usd"] == {"customer": 0.02, "student_est": 0.08, "total": 0.1}
    assert m["student_calls_per_success"] == 20.0
    assert m["student_tokens_per_conversation_median"] == 1000
    assert arm_metrics([rec("1", False)])["student_calls_per_success"] is None  # no pass: no division


# --- The script --------------------------------------------------------------------------------

am = load_script("arms/metrics")


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recs: list[Record],
    *args: str,
    name: str = "A-retail-train",
) -> tuple[int, Path]:
    src = tmp_path / f"{name}.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = tmp_path / "out.metrics.json"
    monkeypatch.setattr(sys, "argv", ["metrics.py", str(src), "--out", str(out), "--owner", "Test", *args])
    return am.main(), out


def test_only_dev_keeps_the_dev_tasks_of_a_file_that_has_both(tmp_path: Path, monkeypatch) -> None:
    """Chiara's telecom file holds τ²-bench's whole train split: our train and dev together."""
    train, dev = splits.task_ids("retail", "train")[:3], splits.task_ids("retail", "dev")[:2]
    recs = [rec(t, False) for t in train] + [rec(t, True) for t in dev]
    code, out = _run(tmp_path, monkeypatch, recs, "--only", "dev")
    m = json.loads(out.read_text())
    assert code == 0 and m["run"] == "arm-A-retail-dev" and (m["passed"], m["conversations"]) == (2, 2)
    assert m["owner"] == "Test" and m["judges"] is True and "dev tasks only" in m["records"]


def test_a_test_split_task_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    recs = [
        rec(splits.task_ids("retail", "train")[0], False),
        rec(splits.task_ids("retail", "test")[0], True),
    ]
    code, out = _run(tmp_path, monkeypatch, recs)
    assert code == 1 and not out.exists() and "test-split" in capsys.readouterr().out


# --- What the script refuses (review of #42) ----------------------------------------------------

TRAIN, DEV = splits.task_ids("retail", "train"), splits.task_ids("retail", "dev")


def test_a_file_with_train_and_dev_tasks_needs_only(tmp_path: Path, monkeypatch, capsys) -> None:
    """Chiara's telecom file is like this: without --only it was summarised as 'train'."""
    code, out = _run(tmp_path, monkeypatch, [rec(TRAIN[0], False), rec(DEV[0], True)])
    assert code == 1 and not out.exists() and "--only" in capsys.readouterr().out


def test_the_split_comes_from_the_records_not_the_file_name(tmp_path: Path, monkeypatch, capsys) -> None:
    code, _ = _run(tmp_path, monkeypatch, [rec(TRAIN[0], False)], name="A-retail-dev")
    assert code == 1 and "named dev, but its tasks are train" in capsys.readouterr().out
    code, out = _run(tmp_path, monkeypatch, [rec(TRAIN[0], False)], name="A-retail-train.retry")
    assert code == 0 and json.loads(out.read_text())["run"] == "arm-A-retail-train"


def test_a_file_mixing_domains_or_arms_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    mixed = [rec(TRAIN[0], False), rec(splits.task_ids("airline", "train")[0], False, domain="airline")]
    assert _run(tmp_path, monkeypatch, mixed)[0] == 1 and "several domains" in capsys.readouterr().out
    arms = [rec(TRAIN[0], False), rec(TRAIN[1], False, arm="C")]
    assert _run(tmp_path, monkeypatch, arms)[0] == 1 and "several arms" in capsys.readouterr().out


def test_a_test_task_is_refused_even_with_only(tmp_path: Path, monkeypatch, capsys) -> None:
    """The filter mustn't hide it: test tasks are never mixed in with train or dev."""
    recs = [rec(DEV[0], True), rec(splits.task_ids("retail", "test")[0], True)]
    code, out = _run(tmp_path, monkeypatch, recs, "--only", "dev")
    assert code == 1 and not out.exists() and "test-split" in capsys.readouterr().out


def test_a_task_in_no_split_is_a_message_not_a_traceback(tmp_path: Path, monkeypatch, capsys) -> None:
    code, _ = _run(tmp_path, monkeypatch, [rec("9999", False)])
    assert code == 1 and "'9999' is in no frozen split" in capsys.readouterr().out


def test_an_existing_summary_is_kept_unless_forced(tmp_path: Path, monkeypatch, capsys) -> None:
    code, out = _run(tmp_path, monkeypatch, [rec(TRAIN[0], True)])
    out.write_text('{"history": "Chiara\'s notes"}')
    code, _ = _run(tmp_path, monkeypatch, [rec(TRAIN[0], True)])
    assert code == 1 and "Chiara" in out.read_text() and "--force" in capsys.readouterr().out
    code, _ = _run(tmp_path, monkeypatch, [rec(TRAIN[0], True)], "--force")
    assert code == 0 and json.loads(out.read_text())["passed"] == 1


def test_a_file_where_everything_crashed_is_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    code, out = _run(tmp_path, monkeypatch, [rec(TRAIN[0], False, error="RateLimitError")])
    assert code == 1 and not out.exists() and "every conversation" in capsys.readouterr().out


def test_the_token_median_ignores_crashed_conversations() -> None:
    crashed = rec("x", False, error="boom", agent_usage={"prompt_tokens": 0, "completion_tokens": 0})
    m = arm_metrics([rec("1", True), rec("2", False), crashed, crashed, crashed])
    assert m["student_tokens_per_conversation_median"] == 1000 and m["conversations"] == 2


def test_several_samples_per_task_get_a_note_on_the_range(tmp_path: Path, monkeypatch) -> None:
    recs = [rec(TRAIN[0], True, sample=0), rec(TRAIN[0], False, sample=1)]
    _, out = _run(tmp_path, monkeypatch, recs)
    m = json.loads(out.read_text())
    assert m["samples_per_task"] == 2 and "too narrow" in m["note_samples"]
