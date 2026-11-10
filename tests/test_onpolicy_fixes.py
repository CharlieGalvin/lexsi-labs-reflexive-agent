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


# --- review fixes -----------------------------------------------------------------------------


def test_at_a_change_never_made_the_last_turns_own_changes_are_still_pending():
    # the last turn cancels R1 and stops: it's the turn rewritten, so R1 must be redone in it
    want = [("cancel_reservation", {"id": "R1"}, None), ("cancel_reservation", {"id": "R2"}, None)]
    view = [SYSTEM, USER, agent(("cancel_reservation", {"id": "R1"})), TOOL, USER]
    point = op.locate(view, want, mutating)
    assert (point.turn, point.kind, point.detail) == (
        2,
        "change_never_made",
        "never done: cancel_reservation",
    )
    assert point.pending == tuple(want)
    both = agent(("cancel_reservation", {"id": "R1"}), ("cancel_reservation", {"id": "R2"}))
    assert op.verify(both, point, mutating) == []


ERROR = {"role": "tool", "tool_call_id": "c0", "content": "Error: reservation not found"}


def test_a_call_the_environment_rejected_isnt_a_change():
    view = [SYSTEM, USER, agent(("cancel_reservation", {"id": "R9"})), ERROR, agent(text="Sorry.")]
    assert op.locate(view, [], mutating) is None  # changed nothing: not an unexpected change
    want = [("cancel_reservation", {"id": "R1"}, None)]
    view = [SYSTEM, USER, agent(("cancel_reservation", {"id": "R1"})), ERROR, USER, agent(text="Done.")]
    point = op.locate(view, want, mutating)  # the expected call failed: the change was never made
    assert point.kind == "change_never_made" and point.pending == tuple(want)


def test_unparsable_arguments_dont_crash():
    msg = {"role": "assistant", "tool_calls": [{"id": "c0", "function": {"name": "x", "arguments": "{bad"}}]}
    assert op.calls_in(msg) == [("x", {"<unparsed>": "{bad"})]


def _val(tmp_path, domain, n, cut=False):
    lines = [json.dumps({"domain": domain, "task_id": str(i)}) for i in range(n)]
    f = tmp_path / f"P-{domain}-val.jsonl"
    f.write_text("\n".join(lines) + ('\n{"domain": "' if cut else "\n"))
    return str(f)


def test_the_validation_exclusion_must_be_whole(tmp_path):
    files = [_val(tmp_path, d, n) for d, n in op.VAL_TASKS.items()]
    ids, problems = op.val_ids(files)
    assert len(ids) == 30 and problems == []
    _, problems = op.val_ids(files[:2])  # telecom missing
    assert problems == ["telecom: 0 validation tasks in --exclude, expected 12"]
    _, problems = op.val_ids([*files[:2], _val(tmp_path, "telecom", 12, cut=True)])
    assert "not valid JSON" in problems[0]  # a cut-short file is an error, not a short read


def test_two_samples_fixed_at_the_same_turn_get_different_ids(monkeypatch):
    from reflexive import heal

    class Teacher:
        def __init__(self, attempts, extra_check):
            self.tried = {}

        def fix(self, failure):
            return {"id": "heal/x", "source": {"teacher": "glm", "attempt": 1}}

    monkeypatch.setattr(heal, "PrefixTeacher", Teacher)
    point = op.FixPoint(5, "unexpected_change", "x", ())

    class F:
        task = ("airline", "7")

        def __init__(self, sample):
            self.record = {"sample": sample}

    ids = {op.teach(F(s), point, mutating, 3)[0]["id"] for s in (0, 1)}
    assert ids == {"onpolicy/glm/airline/7/s0/t5/a1", "onpolicy/glm/airline/7/s1/t5/a1"}
