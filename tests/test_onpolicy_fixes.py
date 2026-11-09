"""The on-policy fix points (scripts/heal/onpolicy_fixes.py): where the student's changes leave the
expected ones, and which teacher turns count as fixing the decision. No τ²-bench needed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "onpolicy_fixes", Path(__file__).parents[1] / "scripts/heal/onpolicy_fixes.py"
)
assert spec is not None and spec.loader is not None
op = importlib.util.module_from_spec(spec)
sys.modules["onpolicy_fixes"] = op  # a dataclass in it needs its module registered
spec.loader.exec_module(op)

WRITES = {"cancel_reservation", "update_reservation_flights", "modify_pending_order_items"}


def mutating(name: str) -> bool:
    return name in WRITES


def agent(*calls: tuple[str, dict], text: str = "") -> dict:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
            for i, (n, a) in enumerate(calls)
        ],
    }


SYSTEM = {"role": "system", "content": "policy"}
USER = {"role": "user", "content": "please"}
TOOL = {"role": "tool", "tool_call_id": "c0", "content": "ok"}


def test_the_first_unneeded_change_is_the_fix_point():
    view = [
        SYSTEM,
        USER,
        agent(("get_reservation_details", {"id": "R1"})),
        TOOL,
        USER,
        agent(("cancel_reservation", {"id": "R1"})),
        TOOL,
    ]
    point = op.locate(view, [], mutating)
    assert (point.turn, point.kind, point.detail) == (
        5,
        "unexpected_change",
        "cancel_reservation: not expected",
    )


def test_the_right_tool_with_wrong_arguments_is_named_so():
    want = [("cancel_reservation", {"id": "R2"}, None)]
    view = [SYSTEM, USER, agent(("cancel_reservation", {"id": "R1"})), TOOL]
    point = op.locate(view, want, mutating)
    assert point.kind == "unexpected_change" and "wrong arguments" in point.detail
    assert point.pending == tuple(want)


def test_a_change_never_made_points_at_the_last_agent_turn():
    want = [("cancel_reservation", {"id": "R1"}, None), ("cancel_reservation", {"id": "R2"}, None)]
    view = [
        SYSTEM,
        USER,
        agent(("cancel_reservation", {"id": "R1"})),
        TOOL,
        USER,
        agent(text="All done, anything else?"),
        USER,
    ]
    point = op.locate(view, want, mutating)
    assert (point.turn, point.kind) == (5, "change_never_made")
    assert point.pending == (want[1],)  # the one already done isn't pending


def test_matching_changes_give_no_fix_point():
    want = [("cancel_reservation", {"id": "R1"}, None)]
    view = [SYSTEM, USER, agent(("cancel_reservation", {"id": "R1"})), TOOL, agent(text="Done.")]
    assert op.locate(view, want, mutating) is None


def test_a_fix_may_hold_back_but_not_make_another_unneeded_change():
    point = op.FixPoint(5, "unexpected_change", "cancel_reservation: not expected", ())
    assert (
        op.verify(agent(text="That fare can't be cancelled under the policy, sorry."), point, mutating) == []
    )
    assert op.verify(agent(("update_reservation_flights", {"id": "R1"})), point, mutating) == [
        "update_reservation_flights matches no expected change still pending"
    ]


def test_at_a_change_never_made_the_fix_must_make_a_pending_change():
    want = ("cancel_reservation", {"id": "R2"}, None)
    point = op.FixPoint(5, "change_never_made", "never done: cancel_reservation", (want,))
    assert op.verify(agent(("cancel_reservation", {"id": "R2"})), point, mutating) == []
    assert op.verify(agent(text="Is there anything else I can help with?"), point, mutating) == [
        "at a change-never-made point the fix must make one of the pending changes"
    ]


def test_pending_at_counts_only_changes_made_before_the_turn():
    want = [("cancel_reservation", {"id": "R1"}, None), ("cancel_reservation", {"id": "R2"}, None)]
    view = [
        SYSTEM,
        USER,
        agent(("cancel_reservation", {"id": "R1"})),
        TOOL,
        agent(("cancel_reservation", {"id": "R2"})),
    ]
    assert op.pending_at(view, 4, want, mutating) == (want[1],)
    assert op.pending_at(view, 2, want, mutating) == tuple(want)


def test_watch_requests_reports_a_retried_request_and_passes_the_reply_on(monkeypatch, capsys):
    httpx = pytest.importorskip("httpx")  # with the teacher's client; CI's dev extra hasn't it

    class Reply:
        status_code, text = 429, '{"error": "rate limited"}'

    monkeypatch.setattr(httpx, "post", lambda *a, **k: Reply())
    op.watch_requests()  # the real (non dry-run) path calls this before any teacher request
    assert httpx.post("https://openrouter.ai/x") is not None
    assert "↻ HTTP 429" in capsys.readouterr().out
