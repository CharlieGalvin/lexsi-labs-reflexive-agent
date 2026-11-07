"""SEL-12: the Tool / Arg / Loop classifier, on real Aya transcripts and on small built cases."""

import json
from pathlib import Path

import pytest

from reflexive.hallucination import FailureType, classify_hallucination, classify_record, domain_tools

CASES = json.loads((Path(__file__).parent / "fixtures" / "hallucination_real.json").read_text())["cases"]
TELECOM, CUSTOMER = domain_tools("telecom")
RETAIL, _ = domain_tools("retail")
AIRLINE, _ = domain_tools("airline")


def classify(messages, tools=TELECOM, **kw):
    return classify_hallucination(messages, tools, customer_tools=CUSTOMER, **kw)


# --- Real transcripts (Arm A / Arm C telecom rollouts) -------------------------------------


@pytest.mark.parametrize("case", [c for c in CASES if c["type"]], ids=lambda c: c["kind"])
def test_real_transcript_finds_its_failure(case):
    d = classify(case["messages"], termination=case["termination"])
    assert any(f.kind == case["kind"] and f.detail == case["detail"] for f in d.findings), d.kinds
    assert FailureType(case["type"]) in d.types


def test_real_dob_given_by_the_customer_is_grounded():
    case = next(c for c in CASES if c["kind"] == "grounded_dob")
    d = classify(case["messages"])
    assert not any(f.kind == "ungrounded" and "dob" in f.detail for f in d.findings)


def test_real_primary_is_the_earliest_finding():
    # The agent's first move: look up "customer's phone number", then call 5 phone tools.
    case = next(c for c in CASES if c["kind"] == "customer_tool")
    d = classify(case["messages"])
    assert d.failure is FailureType.ARG and d.first.kind == "placeholder"
    assert d.kinds["TOOL/customer_tool"] >= 1


# --- Built cases ---------------------------------------------------------------------------

SYS = {"role": "system", "content": "policy"}


def call(name, args, cid="c1", raw=False):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": args if raw else json.dumps(args)},
            }
        ],
    }


def res(content, cid="c1"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def user(text):
    return {"role": "user", "content": text}


def kinds(d):
    return {f.kind for f in d.findings}


def test_clean_conversation_is_none():
    msgs = [
        SYS,
        user("Hi, my number is (555) 123-4567."),
        call("get_customer_by_phone", {"phone_number": "5551234567"}),
        res('{"customer_id": "C1001", "line_ids": ["L1002"]}'),
        call("get_details_by_id", {"id": "L1002"}, "c2"),
        res('{"status": "Active"}', "c2"),
        {"role": "assistant", "content": "Your line is active."},
    ]
    assert classify(msgs).failure is FailureType.NONE


def test_unknown_tool_vs_customer_tool():
    d = classify(
        [
            SYS,
            user("help"),
            call("reset_router", {}),
            res("Error: Tool 'reset_router' not found."),
            call("toggle_airplane_mode", {}, "c2"),
        ]
    )
    assert [f.kind for f in d.findings if f.type is FailureType.TOOL] == ["unknown_tool", "customer_tool"]


def test_schema_violations():
    d = classify(
        [
            SYS,
            user("I'm C1001"),
            call("get_bills_for_customer", {"customer_id": "C1001", "limit": "three", "month": "May"}),
        ]
    )
    assert {"unknown_param", "wrong_type"} <= kinds(d)
    d = classify([SYS, user("I'm C1001"), call("suspend_line", {"customer_id": "C1001"})])
    assert "missing_param" in kinds(d)


def test_numeric_string_is_not_a_type_error():
    d = classify(
        [SYS, user("I'm C1001"), call("get_bills_for_customer", {"customer_id": "C1001", "limit": "3"})]
    )
    assert d.failure is FailureType.NONE


def test_malformed_arguments():
    assert "malformed" in kinds(
        classify([SYS, user("hi"), call("get_customer_by_id", "{customer_id: C1}", raw=True)])
    )


@pytest.mark.parametrize(
    "value",
    [
        "customer_id from previous tool result",
        "<customer_id>",
        "customer_id_from_search",
        "CustomerIDfromPreviousToolResult",
        "not_provided",
        "customer_id",
    ],
)
def test_placeholders(value):
    d = classify([SYS, user("hi"), call("get_customer_by_id", {"customer_id": value})])
    assert d.first.kind == "placeholder"


def test_ungrounded_only_counts_what_came_before():
    later = [
        SYS,
        user("hi"),
        call("get_details_by_id", {"id": "L1002"}),
        res("Error: not found"),
        call("get_customer_by_phone", {"phone_number": "5551234567"}, "c2"),
        res('{"line_ids": ["L1002"]}', "c2"),
    ]
    d = classify(later)
    assert d.first.kind == "ungrounded" and "L1002" in d.first.detail


def test_dob_year_invented():
    d = classify(
        [
            SYS,
            user("I'm Amadou Diallo, born March 14th."),
            call("get_customer_by_name", {"full_name": "Amadou Diallo", "dob": "1990-03-14"}),
        ]
    )
    assert [f.kind for f in d.findings] == ["ungrounded"]
    d = classify(
        [
            SYS,
            user("I'm Amadou Diallo, born March 14th, 1990."),
            call("get_customer_by_name", {"full_name": "amadou diallo", "dob": "1990-03-14"}),
        ]
    )
    assert d.failure is FailureType.NONE


def test_invented_name():
    d = classify(
        [
            SYS,
            user("My MMS won't send."),
            call("get_customer_by_name", {"full_name": "John Smith", "dob": "1985-04-17"}),
        ]
    )
    assert {f.detail.split("=")[0] for f in d.findings} == {"full_name", "dob"}


def test_retail_order_id_with_or_without_hash():
    msgs = [
        SYS,
        user("Cancel order W2378156 please, my email is a.b@x.com"),
        call("find_user_id_by_email", {"email": "a.b@x.com"}),
        res('"mia_garcia_4516"'),
        call("cancel_pending_order", {"order_id": "#W2378156", "reason": "no longer needed"}, "c2"),
    ]
    assert classify(msgs, RETAIL).failure is FailureType.NONE


def test_airline_nested_ids_are_checked():
    msgs = [
        SYS,
        user("Book for me, user sara_doe_496"),
        call(
            "book_reservation",
            {
                "user_id": "sara_doe_496",
                "origin": "JFK",
                "destination": "SEA",
                "flight_type": "one_way",
                "cabin": "economy",
                "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}],
                "passengers": [{"first_name": "Sara", "last_name": "Doe", "dob": "1990-01-01"}],
                "payment_methods": [{"payment_id": "credit_card_1", "amount": 100}],
                "total_baggages": 0,
                "nonfree_baggages": 0,
                "insurance": "no",
            },
        ),
    ]
    details = [f.detail for f in classify(msgs, AIRLINE).findings]
    assert any("HAT136" in x for x in details) and any("credit_card_1" in x for x in details)
    assert not any("JFK" in x or "2024-05-20" in x for x in details)  # worked out, not copied


def test_loops():
    retry = [SYS, user("I'm C1001")]
    for i in range(2):
        retry += [
            call("get_customer_by_id", {"customer_id": "C1001"}, f"c{i}"),
            res("Error: timeout", f"c{i}"),
        ]
    assert classify(retry).failure is FailureType.LOOP and "repeated_failed_call" in kinds(classify(retry))
    same = [SYS, user("I'm C1001")]
    for i in range(3):
        same += [call("get_customer_by_id", {"customer_id": "C1001"}, f"c{i}"), res("{}", f"c{i}")]
    assert "repeated_call" in kinds(classify(same))
    chat = [SYS] + [
        m
        for _ in range(3)
        for m in (user("ok"), {"role": "assistant", "content": "Please take your time to check."})
    ]
    assert "repeated_message" in kinds(classify(chat))
    assert classify([SYS, user("hi")], termination="max_steps").first.kind == "step_limit"


def test_classify_record_uses_the_agents_view():
    rec = {
        "domain": "telecom",
        "termination": "user_stop",
        "messages": [SYS, user("hi")],  # what τ²-bench saved, including the customer's actions
        "agent_view": [SYS, user("hi"), call("check_apn_settings", {})],
    }
    assert classify_record(rec).first.kind == "customer_tool"
    assert classify_record(rec).to_dict()["kinds"] == {"TOOL/customer_tool": 1}


def test_chat_after_a_hand_off_is_a_loop_too():
    # SEL-62: no exemption after transfer_to_human_agents (one dataset's tool name). Chatting on
    # after a hand-off until the step limit is a loop either way, as in Arm A's A56.
    chat = [
        SYS,
        user("help"),
        call("transfer_to_human_agents", {"summary": "no service"}),
        res("Transfer successful"),
    ]
    chat += [
        m
        for _ in range(3)
        for m in (user("ok, waiting"), {"role": "assistant", "content": "You're welcome. Have a good day!"})
    ]
    assert {"repeated_message", "step_limit"} <= kinds(classify(chat, termination="max_steps"))
    claimed = [SYS] + chat[4:]  # says a human is coming, but never handed off: the same
    assert {"repeated_message", "step_limit"} <= kinds(classify(claimed, termination="max_steps"))
    ended = chat[:4] + [{"role": "assistant", "content": "You're welcome. Have a good day!"}]
    assert classify(ended, termination="user_stop").failure is FailureType.NONE  # a clean hand-off


def test_classify_record_takes_the_requests_tools_and_an_optional_customer_label():
    rec = {
        "domain": "telecom",
        "termination": "user_stop",
        "agent_view": [SYS, user("hi"), call("check_apn_settings", {})],
    }
    assert kinds(classify_record(rec)) == {"customer_tool"}  # saved record: the config's lists
    live = [
        {"type": "function", "function": {"name": "get_customer_by_id", "parameters": {"type": "object"}}}
    ]
    assert kinds(classify_record(rec, tools=live)) == {"unknown_tool"}  # the request's tools, no label
    assert kinds(classify_record(rec, tools=live, customer_tools=["check_apn_settings"])) == {"customer_tool"}


def test_example_dot_com_email_is_not_a_placeholder():
    # τ²-bench's retail and airline customers all have @example.com addresses
    msgs = [
        SYS,
        user("My email is yusuf.rossi7301@example.com"),
        call("find_user_id_by_email", {"email": "yusuf.rossi7301@example.com"}),
    ]
    assert classify(msgs, RETAIL).failure is FailureType.NONE
