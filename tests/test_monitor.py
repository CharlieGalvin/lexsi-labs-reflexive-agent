"""SEL-62: the runtime monitor, its guard, the audit log and the AgentTune adapters.

The no_progress tests use the agent turns of the 14 real conversations in
tests/fixtures/no_progress_cases.json (7 loops to the step limit, 7 passes). The AgentTune tests
skip when AgentTune isn't installed (CI); the rest is stdlib.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reflexive.hallucination import domain_tools
from reflexive.monitor import (
    NUDGE,
    AuditWriter,
    Guard,
    RuntimeMonitor,
    audit_records,
    no_progress_turns,
    normalise,
    view_from_audit,
)

CASES = json.loads((Path(__file__).parent / "fixtures" / "no_progress_cases.json").read_text())["cases"]
TOOLS, CUSTOMER = domain_tools("telecom")  # stand-ins for a live request's tools


def view_of(case):
    out = []
    for key, calls in case["agent_turns"]:
        m = {"role": "assistant", "content": key}
        if calls:
            m["tool_calls"] = [
                {
                    "id": f"c{len(out)}_{k}",
                    "type": "function",
                    "function": {"name": "get_customer_by_id", "arguments": "{}"},
                }
                for k in range(calls)
            ]
        out += [m, {"role": "user", "content": "ok"}]
    return out


def say(text):
    return {"role": "assistant", "content": text}


def call(name, args=None, cid="c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}
        ],
    }


SYS = {"role": "system", "content": "policy"}
HI = {"role": "user", "content": "my phone 555-123-4567 has no data"}
BYE = {"role": "user", "content": "Thanks, bye!"}


# --- The no_progress rule, on the real conversations -----------------------------------------


@pytest.mark.parametrize("case", [c for c in CASES if c["group"] == "loops"], ids=lambda c: c["id"])
def test_every_loop_to_the_step_limit_is_caught(case):
    turns = [k for _, k in no_progress_turns(view_of(case))]
    assert case["termination"] == "max_steps" and turns == case["fires_at_agent_turn"] and turns


@pytest.mark.parametrize("case", [c for c in CASES if c["group"] == "passes"], ids=lambda c: c["id"])
def test_no_passing_conversation_sets_it_off(case):
    assert case["passed"] and no_progress_turns(view_of(case)) == []


def test_it_fires_early_on_a56():
    a56 = next(c for c in CASES if c["id"] == "A56")
    assert no_progress_turns(view_of(a56))[0][1] == 44 and len(a56["agent_turns"]) == 100


def test_the_rule_on_small_cases():
    assert normalise("Thank you!") == normalise("thank you.") == "thankyou"
    goodbye = [SYS, HI, say("Glad I could help!"), BYE]
    # three repeats in a row, after the first time: fires on the 4th reply
    loop = goodbye + [
        say("Glad I could help!"),
        BYE,
        say("glad i could help."),
        BYE,
        say("Glad I could help!"),
    ]
    assert no_progress_turns(loop) == [(8, 4)]
    # a tool call in between breaks the run
    broken = goodbye + [say("Glad I could help!"), call("get_customer_by_id"), say("Glad I could help!")]
    assert no_progress_turns(broken) == []
    # repeats spread over a long conversation don't count: they must be in a row
    spread = goodbye + [
        say("Glad I could help!"),
        say("Something new"),
        say("Glad I could help!"),
        say("New again"),
    ]
    assert no_progress_turns(spread) == []


# --- The monitor, turn by turn ----------------------------------------------------------------


def test_watching_turn_by_turn_finds_what_a_whole_read_finds_once_each():
    a56 = view_of(next(c for c in CASES if c["id"] == "A56"))
    mon = RuntimeMonitor(TOOLS)
    seen = []
    for end in range(1, len(a56) + 1):  # after every message
        seen += [a for a in mon.observe(a56[:end]) if a.kind == "no_progress"]
    assert [a.agent_turn for a in seen] == [k for _, k in no_progress_turns(a56)]
    assert len(mon.alarms) == len(set(mon.alarms))  # each reported once


def test_it_uses_the_live_tools_and_customer_tools_are_an_optional_label():
    view = [SYS, HI, call("check_apn_settings")]
    (a,) = RuntimeMonitor(TOOLS).observe(view)  # no dataset knowledge: just not one of its tools
    assert (a.type, a.kind, a.tool, a.agent_turn) == ("TOOL", "unknown_tool", "check_apn_settings", 1)
    (b,) = RuntimeMonitor(TOOLS, customer_tools=CUSTOMER).observe(view)
    assert b.kind == "customer_tool"
    own = [{"type": "function", "function": {"name": "check_apn_settings", "parameters": {"type": "object"}}}]
    assert RuntimeMonitor(own).observe(view) == []  # with that tool in the request, nothing's wrong


# --- The guard --------------------------------------------------------------------------------


def loop_alarms(view):
    mon, out = RuntimeMonitor(TOOLS), []
    for end in range(1, len(view) + 1):
        out.append(mon.observe(view[:end]))
    return out


def test_act_mode_nudges_first_then_stops():
    a68 = view_of(next(c for c in CASES if c["id"] == "A68"))
    guard = Guard("act")
    actions = [(i, act) for i, new in enumerate(loop_alarms(a68)) if (act := guard.react(new))]
    assert [a for _, a in actions] == ["nudge", "stop"]
    assert (guard.nudged_at, guard.stopped_at) == (22, 27)
    assert guard.nudged([SYS])[-1] == {"role": "user", "content": NUDGE}


def test_log_mode_never_changes_the_conversation():
    a68 = view_of(next(c for c in CASES if c["id"] == "A68"))
    guard = Guard("log")
    assert all(guard.react(new) is None for new in loop_alarms(a68))
    assert guard.to_dict() == {"mode": "log", "nudged_at": None, "stopped_at": None}
    with pytest.raises(ValueError, match="mode"):
        Guard("loud")


# --- The audit log ----------------------------------------------------------------------------


def conversation():
    lookup = call("get_customer_by_phone", {"phone_number": "555-123-4567"})
    err = {"role": "tool", "tool_call_id": "c1", "content": "Error: customer not found"}
    return [SYS, HI, lookup, err, say("Could you check the number?"), BYE]


def test_audit_lines_rebuild_the_view_and_mark_failed_calls():
    view = conversation()
    recs = audit_records(view, "t1")
    assert view_from_audit(recs) == {"t1": view}
    (tool,) = [r for r in recs if r["stage_type"] == "tool_call"]
    assert tool["stage_name"] == 'get_customer_by_phone {"phone_number": "555-123-4567"}'
    assert tool["status"] == "error" and tool["error_details"].startswith("Error")
    names = [r["stage_name"] for r in recs if r["stage_type"] != "tool_call"]
    assert len(names) == len(set(names))  # unique: AgentTune's detector never counts them as revisits


def test_a_call_is_logged_only_once_its_result_is_in(tmp_path):
    view, path = conversation(), tmp_path / "audit.jsonl"
    w = AuditWriter(path, "t1")
    for end in range(1, len(view) + 1):  # after every message, the call before its result
        w.write(view[:end])
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    (tool,) = [r for r in recs if r["stage_type"] == "tool_call"]
    assert tool["status"] == "error"  # not "ok", written before the error came back
    w2 = AuditWriter(tmp_path / "crash.jsonl", "t2")
    w2.write(view[:3])  # the call never got a result: held back (what came before is written)...
    assert '"tool_call"' not in (tmp_path / "crash.jsonl").read_text()
    w2.close(view[:3])  # ...until the conversation ends
    (only,) = [
        json.loads(x) for x in (tmp_path / "crash.jsonl").read_text().splitlines() if '"tool_call"' in x
    ]
    assert only["status"] == "ok" and only["tool_name"] == "get_customer_by_phone"


def test_the_writer_appends_only_new_messages_each_turn(tmp_path):
    view, path = conversation(), tmp_path / "audit.jsonl"
    w = AuditWriter(path, "t1")
    w.write(view[:3])
    w.write(view)
    w.write(view)  # nothing new
    recs = [json.loads(line) for line in path.read_text().splitlines()]
    assert view_from_audit(recs) == {"t1": view} and len(recs) == len(audit_records(view, "t1"))


# --- AgentTune --------------------------------------------------------------------------------


def test_tool_call_lines_match_agenttunes_own_audit_records():
    pytest.importorskip("agenttune")
    from reflexive.monitor import event_log

    view = conversation()
    theirs = event_log(view, "t1").to_audit_records()
    ours = [r for r in audit_records(view, "t1") if r["stage_type"] == "tool_call"]
    keys = ("trajectory_id", "stage_name", "tool_name", "stage_type", "state_snapshot", "status")
    assert [{k: r[k] for k in keys} for r in ours] == [{k: r[k] for k in keys} for r in theirs]


def test_alarms_become_agenttune_failures_and_the_detector_reads_the_log(tmp_path):
    pytest.importorskip("agenttune")
    from agenttune.decide.closed_loop.contracts import Failure

    from reflexive.monitor import MonitorDetector, record_event_log, to_failure

    a68 = view_of(next(c for c in CASES if c["id"] == "A68"))
    path = tmp_path / "audit.jsonl"
    AuditWriter(path, "A68").write([SYS, *a68])
    det = MonitorDetector(lambda traj: TOOLS)
    found = list(det.scan_audit_log(str(path)))
    assert all(isinstance(f, Failure) for f in found)
    loops = [f for f in found if f.failure_type == "no_progress"]
    assert len(loops) == len(next(c for c in CASES if c["id"] == "A68")["fires_at_agent_turn"])
    assert list(det.scan_audit_log(str(path))) == []  # polled again: nothing new

    f = to_failure(RuntimeMonitor(TOOLS).observe([SYS, HI, call("check_apn_settings")])[0], "t")
    assert (f.failure_type, f.failed_stage_name) == ("wrong_tool", "check_apn_settings")
    log = record_event_log({"domain": "telecom", "task_id": "x", "agent_view": conversation()})
    assert log.id == "telecom/x/0" and len(log) > 0


# --- The watcher a harness runs after every turn -----------------------------------------------


def test_the_watcher_logs_checks_and_reacts(tmp_path):
    from reflexive.monitor import Watcher

    a68 = [SYS, *view_of(next(c for c in CASES if c["id"] == "A68"))]
    w = Watcher("act", TOOLS, tmp_path / "audit.jsonl", "A68")
    acts = [act for end in range(1, len(a68) + 1) if (act := w.after_turn(a68[:end]))]
    assert acts == ["nudge", "stop"] and (tmp_path / "audit.jsonl").exists()
    d = w.to_dict()
    assert (d["mode"], d["nudged_at"], d["stopped_at"], d["error"]) == ("act", 22, 27, None) and d["alarms"]


def test_a_monitor_error_is_kept_and_never_raised():
    from reflexive.monitor import Watcher

    w = Watcher("log", TOOLS)
    assert w.after_turn([{"role": "assistant", "tool_calls": "not a list"}]) is None
    assert w.error and w.after_turn([SYS]) is None and w.to_dict()["alarms"] == []
    off = Watcher("off", TOOLS)
    assert off.after_turn([SYS, say("hi")]) is None and off.to_dict()["alarms"] == []


def test_the_classifier_stand_in_labels_without_an_llm_and_plugs_into_a_loop():
    pytest.importorskip("agenttune")
    import asyncio
    import types

    from reflexive.monitor import MonitorClassifier, MonitorDetector, plug_into, to_failure

    mon = RuntimeMonitor(TOOLS)
    a68 = [SYS, *view_of(next(c for c in CASES if c["id"] == "A68"))]
    alarms = [a for a in mon.observe(a68) if a.kind in ("no_progress", "missing_param")][:2]
    out = asyncio.run(MonitorClassifier().classify_batch([to_failure(a, "A68") for a in alarms]))
    assert {c.root_cause for c in out} <= {"loop_collapse", "hallucinated_output"} and out[
        0
    ].confidence == 1.0
    loop = plug_into(types.SimpleNamespace(detector=None, classifier=None), lambda t: TOOLS)
    assert isinstance(loop.detector, MonitorDetector) and isinstance(loop.classifier, MonitorClassifier)


def test_a_real_full_closed_loop_ingests_through_our_detector_and_classifier(tmp_path):
    pytest.importorskip("agenttune")
    import asyncio

    from agenttune.decide.closed_loop.full_loop import FullClosedLoop, PathAConfig

    from reflexive.monitor import plug_into

    log = tmp_path / "audit.jsonl"
    a68 = [SYS, *view_of(next(c for c in CASES if c["id"] == "A68"))]
    w = AuditWriter(log, "A68")
    for end in range(1, len(a68) + 1):
        w.write(a68[:end])
    w.close(a68)
    loop = FullClosedLoop(
        PathAConfig(audit_log_path=str(log)),
        retrain_job=lambda *a, **k: None,
        build_model_runner=lambda *a, **k: None,
        old_model_runner=lambda *a, **k: None,
    )
    plug_into(loop, tools_for=lambda traj: TOOLS)
    seen = []

    class Recorder:  # stands in for the LLM-backed TrainingExampleGenerator
        async def generate_batch(self, classified):
            seen.extend(classified)
            return []

    loop.generator = Recorder()
    asyncio.run(loop.ingest_once())
    loops = [c for c in seen if c.failure.failure_type == "no_progress"]
    assert loops and {c.root_cause for c in loops} == {"loop_collapse"}
    assert len(loops) == len(next(c for c in CASES if c["id"] == "A68")["fires_at_agent_turn"])
    n = len(seen)
    asyncio.run(loop.ingest_once())
    assert len(seen) == n  # polled again: nothing new
