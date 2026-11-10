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


# --- review fixes -----------------------------------------------------------------------------


def test_a_saved_pass_with_judges_off_is_unknown():
    assert ev.saved_verdict(rec([], outcome={"passed": True, "skipped": ["NL_ASSERTION"]})) == "unknown"
    assert ev.saved_verdict(rec([], outcome={"passed": True, "skipped": []})) == "success"
    assert ev.saved_verdict(rec([], outcome={"passed": False, "skipped": ["NL_ASSERTION"]})) == "failure"


def test_the_customers_phone_actions_arent_scored_as_the_agents():
    customer_call = {
        "role": "user",
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "toggle_airplane_mode", "arguments": "{}"}}
        ],
    }
    msgs = [
        {"role": "user", "content": "my phone has no signal"},
        customer_call,
        {"role": "tool", "tool_call_id": "c1", "content": "Airplane mode off"},
        customer_call,  # the customer repeats it: not the agent's redundancy
        {"role": "tool", "tool_call_id": "c1", "content": "Airplane mode off"},
        call("get_order", {"order_id": "#1"}),
    ]
    s = ev.agent_scores(rec(msgs), SCHEMAS)
    assert s["tool_call_validity"] == 1.0 and s["redundant_tool_calls"] == 0.0


def test_tool_schemas_need_no_tau2():
    names = [t["function"]["name"] for t in ev.tool_schemas("telecom")]
    assert names and "done" not in names
    r = ev.evaluate_record({"task_id": "1", "messages": []}, ev.saved_verdict)  # no domain: no crash
    assert r["tool_call_validity"] is None


def test_customer_tokens_count_only_when_every_conversation_has_them():
    s = ev.summarise([row("success"), row("failure", user_tokens=None, total_tokens=None)])
    assert s["tokens_counted"] == "agent only"
    assert s["successes_per_million_tokens"] == pytest.approx(1 / 2000 * 1e6, rel=1e-3)
    assert ev.summarise([row("success")])["tokens_counted"] == "agent and customer"


def test_no_efficiency_without_a_success_count():
    s = ev.summarise([row("success"), row("unknown")])
    assert s["successes_per_million_tokens"] is None and s["successes_per_dollar"] is None


def test_crashed_spend_counts_in_the_cost():
    s = ev.summarise([row("success")], crashed=1, crashed_cost=0.04)
    assert s["cost_usd_total"] == pytest.approx(0.05) and s["successes_per_dollar"] == 20.0


def test_evaluate_keeps_the_crashes_spend(monkeypatch):
    monkeypatch.setattr(ev, "evaluate_record", lambda r, success, judge: row(success(r), task_id="retail:1"))
    crash = rec([], error="crash", task_id="2", agent_usage={"est_cost": 0.03}, user_cost=0.01)
    _, s = ev.evaluate([rec([], outcome={"passed": True}), crash], ev.saved_verdict)
    assert s["crashed_not_scored"] == 1 and s["crashed_cost_usd"] == pytest.approx(0.04)


# --- the script -------------------------------------------------------------------------------


def _script():
    from tests._scripts import load_script

    return load_script("arms/evaluate")


def test_a_task_in_no_split_is_unknown_not_a_crash():
    from reflexive import splits

    sc = _script()
    dev = splits.task_ids("retail", "dev")[0]
    assert sc.split_of([{"domain": "retail", "task_id": dev}]) == "dev"
    assert sc.split_of([{"domain": "retail", "task_id": "no-such-task"}]) == "unknown"


def _file(arm, task, sample=0, **kw):
    r = {"domain": "retail", "task_id": task, "sample": sample, "arm": arm, "outcome": {"passed": True}, **kw}
    return r, row("success", task_id=f"retail:{task}", sample=sample)


def test_overall_refuses_different_arms():
    sc = _script()
    (a, ra), (s, rs) = _file("A", "1"), _file("S", "1")
    got, why = sc.overall([([a], [ra], "dev"), ([s], [rs], "dev")])
    assert got is None and "different arms" in why


def test_overall_refuses_different_splits():
    sc = _script()
    (a, ra), (b, rb) = _file("A", "1"), _file("A", "2")
    got, why = sc.overall([([a], [ra], "dev"), ([b], [rb], "test")])
    assert got is None and "splits" in why


def test_overall_counts_a_conversation_in_two_files_once():
    sc = _script()
    crash, _ = _file("A", "1", error="rate limit")
    rerun, rr = _file("A", "1")
    other, ro = _file("A", "2")
    got, _ = sc.overall([([crash, other], [ro], "dev"), ([rerun], [rr], "dev")])
    assert got["conversations"] == 2 and got["crashed_not_scored"] == 0
