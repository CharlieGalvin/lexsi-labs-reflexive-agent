"""SEL-38's per-conversation metrics and the arm summary (no τ²-bench calls)."""

import pytest

from reflexive import evaluate as ev

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    }
]


def call(name, args):
    import json

    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def rec(messages, **kw):
    return {"domain": "retail", "task_id": "1", "messages": messages, "outcome": {"passed": False}, **kw}


def test_validity_and_redundancy_count_calls():
    msgs = [
        {"role": "user", "content": "where is my order?"},
        call("get_order", {"order_id": "#1"}),
        call("get_order", {"order_id": "#1"}),  # a repeat
        call("made_up_tool", {}),  # not offered
    ]
    s = ev.agent_scores(rec(msgs), SCHEMAS)
    assert s["tool_call_validity"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["redundant_tool_calls"] == pytest.approx(1 / 3, abs=1e-3)


def test_goodbye_loop_is_a_loop():
    bye = {"role": "assistant", "content": "You're welcome! Have a great day!"}
    msgs = [{"role": "user", "content": "thanks"}, bye, {"role": "user", "content": "bye"}, bye]
    s = ev.agent_scores(rec(msgs), SCHEMAS)
    assert s["agent_loop_detection"] == 0.0 and "stagnation" in s["loop_signals"]


def test_no_calls_means_no_validity_score():
    s = ev.agent_scores(
        rec([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"}]), SCHEMAS
    )
    assert s["tool_call_validity"] is None and s["redundant_tool_calls"] is None


def test_ttc_adds_the_customer_when_recorded():
    r = rec(
        [],
        agent_usage={"prompt_tokens": 100, "completion_tokens": 10, "est_cost": 0.01, "calls": 2},
        user_usage={"prompt_tokens": 50, "completion_tokens": 5},
        user_cost=0.002,
        wall_s=12.0,
    )
    t = ev.ttc(r)
    assert (t["agent_tokens"], t["user_tokens"], t["total_tokens"]) == (110, 55, 165)
    assert t["cost_usd"] == pytest.approx(0.012) and t["seconds"] == 12.0
    old = ev.ttc(rec([], agent_usage={"prompt_tokens": 100, "completion_tokens": 10}))
    assert old["user_tokens"] is None and old["total_tokens"] is None


def row(verdict, **kw):
    base = {
        "verdict": verdict,
        "tool_call_validity": 1.0,
        "redundant_tool_calls": 0.0,
        "agent_loop_detection": 1.0,
        "agent_tokens": 1000,
        "user_tokens": 100,
        "total_tokens": 1100,
        "cost_usd": 0.01,
        "seconds": 10.0,
        "steps": 5,
    }
    return {**base, **kw}


def test_summary_rate_range_and_efficiency():
    s = ev.summarise(
        [row("success"), row("failure", agent_loop_detection=0.0, tool_call_validity=0.5)], crashed=1
    )
    assert s["success_rate"] == 0.5 and s["crashed_not_scored"] == 1
    assert s["conversations_looping"] == 1 and s["conversations_with_invalid_call"] == 1
    assert s["successes_per_million_tokens"] == pytest.approx(1 / 2200 * 1e6, rel=1e-3)
    assert s["successes_per_dollar"] == 50.0


def test_unknown_verdicts_give_bounds_not_a_rate():
    s = ev.summarise([row("success"), row("failure"), row("unknown")])
    assert s["success_rate"] is None and s["success_bounds"] == [
        pytest.approx(1 / 3, abs=1e-3),
        pytest.approx(2 / 3, abs=1e-3),
    ]


def test_an_oracle_error_gives_bounds_not_a_rate():
    s = ev.summarise([row("success"), row("failure"), row("error")])
    assert s["success_rate"] is None and s["success_ci95"] is None
    assert s["success_bounds"] == [pytest.approx(1 / 3, abs=1e-3), pytest.approx(2 / 3, abs=1e-3)]


def test_the_range_resamples_tasks_not_conversations():
    # 10 tasks, each passed in both samples or failed in both: 20 conversations, 10 independent results
    rows = [
        row("success" if t < 5 else "failure", task_id=f"retail:{t}", sample=s)
        for t in range(10)
        for s in (0, 1)
    ]
    lo, hi = ev.summarise(rows)["success_ci95"]
    by_conv = [row("success" if i < 10 else "failure", task_id=f"retail:{i}") for i in range(20)]
    lo2, hi2 = ev.summarise(by_conv)["success_ci95"]
    assert hi - lo > hi2 - lo2  # paired samples carry less information than 20 separate tasks


def test_each_sample_has_its_rate():
    rows = [row("success", task_id="r:1", sample=0), row("failure", task_id="r:2", sample=0)]
    rows += [row("success", task_id="r:1", sample=1), row("success", task_id="r:2", sample=1)]
    assert ev.summarise(rows)["success_by_sample"] == {"rates": {"0": 0.5, "1": 1.0}, "mean": 0.75}


def test_evaluate_keeps_the_rerun(monkeypatch):
    monkeypatch.setattr(ev, "evaluate_record", lambda r, success, judge: row(success(r), task_id="retail:1"))
    a = rec([], error="crash")
    b = rec([], outcome={"passed": True})
    rows, s = ev.evaluate([a, b], ev.saved_verdict)
    assert [r["verdict"] for r in rows] == ["success"] and s["crashed_not_scored"] == 0
