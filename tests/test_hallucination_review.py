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
