"""SEL-27: the τ²-bench comparison metrics and their bootstrap confidence intervals."""

from __future__ import annotations

import json

import pytest

from reflexive.compare import (
    METRICS,
    bootstrap,
    compare,
    conversation_values,
    paired_bootstrap_diff,
    redundant_calls,
    steps,
    to_markdown,
)

SYS = {"role": "system", "content": "policy"}
HI = {"role": "user", "content": "my phone 555-123-4567 has no data"}


def call(name, args=None, cid="c"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}
        ],
    }


def result(cid="c", text="ok"):
    return {"role": "tool", "tool_call_id": cid, "content": text}


LOOKUP = call("get_customer_by_phone", {"phone_number": "555-123-4567"})
DONE = {"role": "assistant", "content": "All set."}


def rec(task, passed, view, domain="telecom", **kw):
    return {"domain": domain, "task_id": task, "termination": "user_stop", "outcome": {"passed": passed},
            "messages": view, "agent_view": view, **kw}  # fmt: skip


# --- Per conversation -------------------------------------------------------------------------


def test_redundant_calls_are_exact_repeats_whatever_the_key_order():
    again = call("get_customer_by_phone", {"phone_number": "555-123-4567"}, cid="d")
    other = call("get_customer_by_phone", {"phone_number": "555-000-0000"}, cid="e")
    view = [SYS, HI, LOOKUP, result(), again, result("d"), other, result("e"), DONE]
    assert redundant_calls(rec("t", True, view)) == (1, 3)
    assert steps(rec("t", True, view)) == 4  # three tool-call turns and the reply


def test_the_values_use_the_classifier():
    wrong_tool = call("check_apn_settings")  # the customer's phone tool: TOOL
    v = conversation_values(rec("t", False, [SYS, HI, wrong_tool]))
    assert v["success_rate"] == (0.0, 1.0) and v["hallucination_rate"] == (1.0, 1.0)
    assert v["loop_rate"] == (0.0, 1.0) and v["redundant_call_rate"] == (0.0, 1.0)
    clean = conversation_values(rec("t", True, [SYS, HI, LOOKUP, result(), DONE]))
    assert clean["hallucination_rate"] == (0.0, 1.0) and clean["success_rate"] == (1.0, 1.0)


# --- The bootstrap ----------------------------------------------------------------------------


def test_a_bootstrap_range_contains_the_value_and_is_repeatable():
    pairs = [(1.0, 1.0)] * 30 + [(0.0, 1.0)] * 70
    ci = bootstrap(pairs, reps=2000)
    assert ci is not None
    lo, hi = ci
    assert lo < 0.3 < hi and 0.15 < lo and hi < 0.45
    assert bootstrap(pairs, reps=2000) == (lo, hi)  # same seed, same range
    assert bootstrap([(1.0, 1.0)] * 10, reps=200) == (1.0, 1.0)  # no spread, no range
    assert bootstrap([]) is None


def test_a_ratio_metric_pools_numerators_and_denominators():
    # 1 repeat in 10 calls, then 0 in 1: pooled 1/11, not the mean of 0.1 and 0
    ci = bootstrap([(1.0, 10.0), (0.0, 1.0)], reps=2000)
    assert ci is not None
    lo, hi = ci
    assert 0.0 <= lo <= 1 / 11 <= hi <= 0.1


def test_the_paired_difference_sees_a_change_on_the_same_tasks():
    tasks = [("telecom", str(i)) for i in range(40)]
    a = {t: [(float(i < 10), 1.0)] for i, t in enumerate(tasks)}  # 25%
    b = {t: [(float(i < 30), 1.0)] for i, t in enumerate(tasks)}  # 75%: 20 more, none fewer
    ci = paired_bootstrap_diff(a, b, reps=2000)
    assert ci is not None
    lo, hi = ci
    assert 0.3 < lo <= 0.5 <= hi < 0.7
    assert paired_bootstrap_diff(a, {("retail", "x"): [(1.0, 1.0)]}) is None  # nothing shared


def test_samples_of_one_task_stay_together():
    # two samples per task in b: the task, not each conversation, is resampled
    a = {("d", str(i)): [(0.0, 1.0)] for i in range(20)}
    b = {("d", str(i)): [(1.0, 1.0), (0.0, 1.0)] for i in range(20)}
    assert paired_bootstrap_diff(a, b, reps=500) == (0.5, 0.5)


# --- Comparing two runs -----------------------------------------------------------------------


def runs():
    tasks = [str(i) for i in range(30)]
    ok = [SYS, HI, LOOKUP, result(), DONE]
    bad = [SYS, HI, call("check_apn_settings")]
    a = [rec(t, i < 5, ok if i < 5 else bad) for i, t in enumerate(tasks)]
    b = [rec(t, i < 25, ok if i < 25 else bad) for i, t in enumerate(tasks)]
    b.append(rec("crashed", False, [], error="RateLimitError"))  # left out, counted apart
    return a, b


def test_compare_gives_every_metric_with_ranges_and_marks_clear_differences():
    a, b = runs()
    c = compare(a, b, reps=2000)
    assert [r.metric for r in c.rows] == list(METRICS)
    assert (c.a_scored, c.b_scored, c.b_crashed, c.shared_tasks) == (30, 30, 1, 30)
    sr = c.rows[0]
    assert sr.a == pytest.approx(5 / 30) and sr.b == pytest.approx(25 / 30)
    assert sr.diff == pytest.approx(20 / 30) and sr.clear
    hall = c.rows[1]
    assert hall.diff == pytest.approx(-20 / 30) and hall.clear
    assert not c.rows[2].clear  # no loops either side: no difference

    md = to_markdown(c, "A", "B")
    assert "| Success rate | 16.7% |" in md and "+66.7 pts" in md
    assert "B better" in md.splitlines()[2] and "B better" in md.splitlines()[3]  # fewer hallucinations
    assert "(0 and 1 crashed, left out)" in md
    assert json.loads(json.dumps(c.to_dict()))["rows"][0]["clear"] is True


def test_a_run_with_nothing_scored_is_refused():
    a, _ = runs()
    with pytest.raises(ValueError, match="no scored"):
        compare(a, [rec("x", False, [], error="boom")])


def test_the_guards_stops_are_shown_on_their_own():
    # SEL-69: the ablation's gain shouldn't be partly the harness ending conversations
    a, b = runs()
    b = [{**r, "termination": "monitor_stop"} if r["task_id"] in ("0", "29") else r for r in b]
    c = compare(a, b, reps=200)
    assert c.a_monitor_stopped is not None and c.a_monitor_stopped["conversations"] == 0
    assert c.b_monitor_stopped == {
        "conversations": 2,
        "passed": 1,
        "task_success_if_failed": round(24 / 30, 4),
    }
    md = to_markdown(c, "A", "A-guard")
    assert "A-guard: the runtime monitor's guard ended 2 conversations" in md and "1 of them passed" in md
    assert "A: the runtime monitor" not in md  # nothing to say for a run the guard never stopped
    assert json.loads(json.dumps(c.to_dict()))["b_monitor_stopped"]["passed"] == 1
