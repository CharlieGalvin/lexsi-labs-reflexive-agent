"""Regression tests from the #40 review (5 Nov): each one failed on the first version of the classifier."""

from __future__ import annotations

import json
from typing import Any

from reflexive.hallucination import (
    agent_view_of,
    classify_hallucination,
    classify_record,
    domain_tools,
)

Msg = dict[str, Any]
TOOLS, CUSTOMER = domain_tools("telecom")
SYSTEM: Msg = {"role": "system", "content": "You are a telecom support agent."}


def user(text: str) -> Msg:
    return {"role": "user", "content": text}


def call(cid: str, name: str, args: dict[str, Any]) -> Msg:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        ],
    }


def result(cid: str, text: str) -> Msg:
    return {"role": "tool", "tool_call_id": cid, "content": text}


def kinds(messages: list[Msg]) -> list[tuple[str, str]]:
    return [
        (str(f.type), f.kind)
        for f in classify_hallucination(messages, TOOLS, customer_tools=CUSTOMER).findings
    ]


# --- A record without agent_view is judged on what the agent saw --------------------------------


def _record_without_agent_view() -> dict[str, Any]:
    phone_action = {
        "role": "user",
        "content": None,
        "tool_calls": [
            {"id": "u1", "type": "function", "function": {"name": "check_sim_status", "arguments": "{}"}}
        ],
    }
    return {
        "domain": "telecom",
        "termination": "user_stop",
        "messages": [
            user("My customer ID is C1."),
            phone_action,
            result("u1", "SIM active on line L1009"),  # the customer's phone: the agent never sees this
            user("I checked my phone."),
            call("a1", "get_data_usage", {"customer_id": "C1", "line_id": "L1009"}),
        ],
    }


def test_record_without_agent_view_drops_the_customers_phone_actions() -> None:
    view = agent_view_of(_record_without_agent_view())
    assert not any(m["role"] == "user" and m.get("tool_calls") for m in view)
    assert all("L1009" not in (m.get("content") or "") for m in view)


def test_record_without_agent_view_still_catches_an_invented_line_id() -> None:
    found = [(str(f.type), f.kind, f.detail) for f in classify_record(_record_without_agent_view()).findings]
    assert any(k == "ungrounded" and "L1009" in d for _, k, d in found), found


# --- "Seen before" means the whole value, not a fragment of something else -----------------------


def test_a_truncated_id_is_not_grounded_by_a_longer_one() -> None:
    msgs = [
        SYSTEM,
        user("my number is 5551234567"),
        call("a", "get_customer_by_phone", {"phone_number": "5551234567"}),
        result("a", '{"customer_id":"C1001","line_ids":["L1001"]}'),
        call("b", "get_data_usage", {"customer_id": "C100", "line_id": "L10"}),
    ]
    assert kinds(msgs).count(("ARG", "ungrounded")) == 2


def test_an_invented_name_is_not_grounded_by_parts_of_other_words() -> None:
    msgs = [
        SYSTEM,
        user("Every customer on this channel plan. DOB 1990-01-01"),
        call("a", "get_customer_by_name", {"full_name": "Tom Ann", "dob": "1990-01-01"}),
    ]  # "cus-tom-er", "ch-ann-el"
    assert ("ARG", "ungrounded") in kinds(msgs)


def test_whole_values_still_ground() -> None:
    msgs = [
        SYSTEM,
        user("I'm John Smith, dob 1990-01-01, my email is john.smith@example.com."),
        call("a", "get_customer_by_name", {"full_name": "Smith John", "dob": "1990-01-01"}),
        result("a", '[{"customer_id":"C1001","line_ids":["L1001"]}]'),
        call("b", "get_data_usage", {"customer_id": "C1001", "line_id": "L1001"}),
        user("order #W123 please"),
        call("c", "get_details_by_id", {"id": "W123"}),
    ]
    assert ("ARG", "ungrounded") not in kinds(msgs)


# --- A retry means repeating a failure the agent had already seen ----------------------------


def _two_calls(ids: tuple[str, str], name: str, args: dict[str, Any]) -> Msg:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
            for i in ids
        ],
    }


def test_two_identical_calls_in_one_turn_are_not_a_retry() -> None:
    msgs = [
        SYSTEM,
        user("my id is C9"),
        _two_calls(("x", "y"), "get_customer_by_id", {"customer_id": "C9"}),
        result("x", "Error: Customer not found"),
        result("y", "Error: Customer not found"),
    ]
    assert ("LOOP", "repeated_failed_call") not in kinds(msgs)


def test_a_retry_in_a_later_turn_is_caught_and_names_the_turn_it_repeats() -> None:
    msgs = [
        SYSTEM,
        user("my id is C9"),
        call("x", "get_customer_by_id", {"customer_id": "C9"}),
        result("x", "Error: Customer not found"),
        user("please try again"),
        call("y", "get_customer_by_id", {"customer_id": "C9"}),
        result("y", "Error: Customer not found"),
    ]
    found = classify_hallucination(msgs, TOOLS, customer_tools=CUSTOMER).findings
    retry = [f for f in found if f.kind == "repeated_failed_call"]
    assert len(retry) == 1 and retry[0].turn == 5 and "at turn 2" in retry[0].detail


# --- A real phone number with spaces is not a placeholder -------------------------------------


def test_a_phone_number_written_with_spaces_is_not_a_placeholder() -> None:
    for written in ("(555) 123-4567", "555 123 4567"):
        msgs = [
            SYSTEM,
            user(f"My number is {written}."),
            call("a", "get_customer_by_phone", {"phone_number": written}),
        ]
        assert kinds(msgs) == [], written


def test_a_described_phone_number_is_still_a_placeholder() -> None:
    msgs = [
        SYSTEM,
        user("hi"),
        call("a", "get_customer_by_phone", {"phone_number": "customer's phone number"}),
    ]
    assert ("ARG", "placeholder") in kinds(msgs)


# --- Names without accents and year-first dates ground -------------------------------------------


def test_a_name_written_without_its_accents_is_grounded() -> None:
    msgs = [
        SYSTEM,
        user("I'm José García, born 1990-03-14"),
        call("a", "get_customer_by_name", {"full_name": "Jose Garcia", "dob": "1990-03-14"}),
    ]
    assert kinds(msgs) == []


def test_a_year_first_date_of_birth_is_grounded() -> None:
    msgs = [
        SYSTEM,
        user("I'm John Smith, dob 1990/03/14"),
        call("a", "get_customer_by_name", {"full_name": "John Smith", "dob": "1990-03-14"}),
    ]
    assert kinds(msgs) == []


# --- The saved tool schemas still match τ²-bench (only where τ²-bench is installed) --------------


def test_saved_tool_schemas_match_tau2_bench() -> None:
    import subprocess
    import sys
    from pathlib import Path

    import pytest

    pytest.importorskip("tau2")
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "scripts/dump_tool_schemas.py", "--check"], cwd=root, capture_output=True, text=True
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_spot_check_reports_a_missing_file_instead_of_crashing(tmp_path: Any) -> None:
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    run = subprocess.run(
        [sys.executable, "scripts/spot_check_hallucination.py", str(tmp_path / "nope.jsonl")],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1 and "not found" in run.stdout and "Traceback" not in run.stderr
