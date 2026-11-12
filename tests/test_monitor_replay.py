"""SEL-69: saved rollouts replayed through the runtime monitor, turn by turn."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/arms/monitor_replay.py"
spec = importlib.util.spec_from_file_location("monitor_replay", SCRIPT)
assert spec is not None and spec.loader is not None
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)


def say(text):
    return {"role": "assistant", "content": text}


def record(view, termination, passed, task="1"):
    return {
        "arm": "A",
        "domain": "retail",
        "task_id": task,
        "termination": termination,
        "outcome": {"passed": passed},
        "agent_view": view,
    }


HI = {"role": "user", "content": "hi"}
BYE = {"role": "user", "content": "bye"}
LOOP = [HI, say("Done!"), BYE, *[m for _ in range(6) for m in (say("Glad to help, goodbye!"), BYE)]]
CLEAN = [HI, say("Which order?"), {"role": "user", "content": "W1"}, say("Cancelled."), BYE]


def test_a_goodbye_loop_fires_and_is_nudged_as_in_a_live_run():
    row = mr.replay(record(LOOP, "max_steps", False))
    # agent turns: 1 "Done!", 2 first goodbye, 3-5 repeats → fires at 5 (6-7 are only 2 more)
    assert row["fires_at_agent_turn"] == [5]
    assert (row["nudged_at"], row["stopped_at"]) == (5, None)
    assert row["longest_repeat_run"] == 5


def test_verdicts_flag_false_alarms_and_misses():
    base = {"fires_at_agent_turn": [], "termination": "user_stop", "passed": True}
    assert mr.verdict(base) == "quiet"
    assert mr.verdict({**base, "fires_at_agent_turn": [5]}) == "false alarm: passed"
    assert mr.verdict({**base, "passed": False, "fires_at_agent_turn": [5]}) == "fired, not a step-limit loop"
    assert mr.verdict({**base, "passed": False, "termination": "max_steps"}) == "missed: step-limit, no fire"
    assert (
        mr.verdict({**base, "passed": False, "termination": "max_steps", "fires_at_agent_turn": [5]})
        == "caught"
    )


def test_main_writes_the_summary(tmp_path):
    f = tmp_path / "A-retail-dev.jsonl"
    recs = [record(LOOP, "max_steps", False, "1"), record(CLEAN, "user_stop", True, "2")]
    f.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = tmp_path / "m.metrics.json"
    assert mr.main([str(f), "--metrics", str(out)]) == 0
    s = json.loads(out.read_text())["files"]["A-retail-dev.jsonl"]
    assert s["conversations"] == 2 and s["step_limit"] == 1 and s["passed"] == 1
    assert s["verdicts"] == {"caught": 1, "quiet": 1}
    assert [c["id"] for c in s["fired"]] == ["A0"]
