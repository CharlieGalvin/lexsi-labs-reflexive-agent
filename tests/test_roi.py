"""scripts/arms/roi.py (SEL-86, SEL-108): latency, cost and success from saved records. Synthetic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _scripts import load_script

roi = load_script("arms/roi")


def rec(passed: bool, seconds: float, calls: int, pt: int = 0, ct: int = 0, user: float = 0.0, **kw) -> dict:
    return {
        "outcome": {"passed": passed, "skipped": kw.get("skipped", [])},
        "termination": "user_stop",
        "agent_usage": {"seconds": seconds, "calls": calls, "prompt_tokens": pt, "completion_tokens": ct},
        "user_cost": user,
    }


def write(path: Path, recs: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_api_and_self_hosted_costs(tmp_path: Path) -> None:
    # GPT: 1M input + 0.1M output tokens over 2 conversations = $2 + $0.80; customer $0.20; 1 solved
    write(
        tmp_path / "GPT-retail-test.jsonl",
        [rec(True, 10, 5, 500_000, 50_000, 0.1), rec(False, 30, 5, 500_000, 50_000, 0.1)],
    )
    # S2: 3,600 s of model time on a $4/h GPU shared by 4 = $1; customer $1; 2 solved
    write(
        tmp_path / "S2-retail-test.jsonl",
        [rec(True, 1800, 10, user=0.5), rec(True, 1800, 30, user=0.5), {**rec(True, 1, 1), "error": "boom"}],
    )  # a crash: left out
    out = tmp_path / "roi.json"
    assert (
        roi.main(
            [
                "GPT",
                "S2",
                "--episodes",
                str(tmp_path),
                "--gpu-hourly",
                "4",
                "--concurrency",
                "4",
                "--json",
                str(out),
            ]
        )
        == 0
    )
    rows = json.loads(out.read_text())["rows"]
    g = rows["GPT/retail"]
    assert g["cost_per_conversation_usd"] == pytest.approx((2.0 + 0.8 + 0.2) / 2)
    assert g["cost_per_solved_task_usd"] == pytest.approx(3.0)
    assert g["seconds_per_request_median"] == pytest.approx(4.0)  # 2 s and 6 s per request
    assert g["seconds_per_request_pooled"] == pytest.approx(4.0)
    s = rows["S2/retail"]
    assert s["conversations"] == 2 and s["success"] == 1.0
    assert s["agent_cost_usd"] == pytest.approx(1.0)
    assert s["cost_per_solved_task_usd"] == pytest.approx(1.0)
    assert s["seconds_per_request_mean"] == pytest.approx((180 + 60) / 2)
    assert s["seconds_per_request_pooled"] == pytest.approx(3600 / 40)
    assert rows["S2/all"]["conversations"] == 2  # domains without records are skipped


def test_self_hosted_without_a_gpu_price_has_no_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write(tmp_path / "S2-airline-test.jsonl", [rec(False, 10, 2, user=0.1)])
    assert roi.main(["S2", "--episodes", str(tmp_path)]) == 0
    assert "no --gpu-hourly" in capsys.readouterr().out
