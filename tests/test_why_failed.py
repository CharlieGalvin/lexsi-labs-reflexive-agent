"""SEL-75: `just arm-why`: what failed conversations left undone, and what one arm lost against another.

The re-scoring is τ²-bench's (`reflexive.outcome.score`); here it's replaced by fixed details, so
the test checks the counting and the pairing, not the evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

wf = load_script("arms/why_failed")


def rec(
    task: str, sample: int, passed: bool, termination: str = "user_stop", customer_calls: int = 0
) -> dict[str, Any]:
    msgs = [{"role": "user", "content": "", "tool_calls": [{"id": f"u{i}"} for i in range(customer_calls)]}]
    return {
        "task_id": task,
        "sample": sample,
        "termination": termination,
        "messages": msgs,
        "outcome": {"passed": passed},
    }


def write(folder: Path, arm: str, rows: list[dict[str, Any]]) -> None:
    (folder / f"{arm}-telecom-dev.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_counts_unmet_assertions_and_lists_what_the_second_arm_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    write(tmp_path, "S2", [rec("t1", 0, True, customer_calls=3), rec("t2", 0, False), rec("t3", 0, True)])
    write(
        tmp_path,
        "H",
        [rec("t1", 0, False, customer_calls=1), rec("t2", 0, False), rec("t3", 0, False, "max_steps")],
    )
    details = {
        "t1": {
            "unmet": ["assert_mobile_data_status(expected_status=true)"],
            "missed": ["user: toggle_airplane_mode()"],
            "db_match": None,
            "checks": {},
        },
        "t2": {
            "unmet": ["assert_can_send_mms(expected_status=true)"],
            "missed": [],
            "db_match": None,
            "checks": {},
        },
    }
    monkeypatch.setattr(wf, "detail", lambda domain, r: details[r["task_id"]])
    assert wf.main(["S2", "H", "--dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1  assert_can_send_mms(expected_status=true)" in out  # S2's one normal failure
    assert "# 2 conversations S2 passed and H failed" in out
    assert "not met: assert_mobile_data_status(expected_status=true)" in out
    assert "missed:  user: toggle_airplane_mode()" in out
    assert "customer actions 1 (S2 3)" in out
    assert "ended abnormally: no assertion detail" in out  # t3 hit the step limit


def test_a_missing_file_is_named(tmp_path: Path):
    with pytest.raises(SystemExit, match="no .*X-telecom-dev.jsonl"):
        wf.main(["X", "--dir", str(tmp_path)])
