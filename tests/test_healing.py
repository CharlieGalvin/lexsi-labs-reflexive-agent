from reflexive.healing import FailureType, classify_hallucination


def get_orders(user_id: str, limit: int = 3):
    """Last orders for a user."""
    return []


def send_email(to: str, body: str):
    """Send an email."""
    return "sent"


TOOLS = [get_orders, send_email]
PROMPT = [{"role": "user", "content": "Email the manager of user u_1042 about their failed orders."}]


def call(name, args, cid="c1"):
    import json
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def res(content, cid="c1"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def test_tool_not_found():
    d = classify_hallucination([call("fetch_orders", {"user_id": "u_1042"})], TOOLS, PROMPT)
    assert d.failure is FailureType.TOOL and d.step == 0


def test_arg_invented_id():
    d = classify_hallucination([call("get_orders", {"user_id": "u_9999"}), res("[]")], TOOLS, PROMPT)
    assert d.failure is FailureType.ARG and "u_9999" in d.reasons[0]


def test_arg_grounded_in_prior_result_is_fine():
    traj = [call("get_orders", {"user_id": "u_1042"}), res('[{"manager": "ana@corp.com"}]'),
            call("send_email", {"to": "ana@corp.com", "body": "Summary"}, "c2"), res("sent", "c2"),
            {"role": "assistant", "content": "Done."}]
    assert classify_hallucination(traj, TOOLS, PROMPT).failure is FailureType.NONE


def test_arg_email_invented_before_it_was_seen():
    traj = [call("send_email", {"to": "ana@corp.com", "body": "x"}), res("sent"),
            call("get_orders", {"user_id": "u_1042"}, "c2"), res('[{"manager": "ana@corp.com"}]', "c2")]
    assert classify_hallucination(traj, TOOLS, PROMPT).failure is FailureType.ARG


def test_arg_schema_violation():
    d = classify_hallucination([call("get_orders", {"customer": "u_1042"})], TOOLS, PROMPT)
    assert d.failure is FailureType.ARG and any("unknown parameter" in r for r in d.reasons)


def test_loop():
    traj = []
    for i in range(3):
        traj += [call("get_orders", {"user_id": "u_1042"}, f"c{i}"), res("[]", f"c{i}")]
    assert classify_hallucination(traj, TOOLS, PROMPT).failure is FailureType.LOOP


def test_budget_exhausted_is_loop():
    traj = [call("get_orders", {"user_id": "u_1042"}), res("[]"),
            call("get_orders", {"user_id": "u_1042", "limit": 5}, "c2"), res("[]", "c2")]
    assert classify_hallucination(traj, TOOLS, PROMPT, max_steps=2).failure is FailureType.LOOP
