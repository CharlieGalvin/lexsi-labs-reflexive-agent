import pytest

from reflexive import toy_lookup as tl


def test_database_is_fixed_and_names_unique():
    db = tl.customers()
    assert len(db) == tl.N_CUSTOMERS
    assert len({c["name"] for c in db}) == tl.N_CUSTOMERS
    assert db[0] == tl.customers.__wrapped__()[0]  # same seed, same data


def test_tools():
    c = tl.customers()[5]
    assert tl.find_customer(c["name"].upper()) == {"customer_id": c["customer_id"]}
    assert tl.get_customer(c["customer_id"].lower())["balance"] == c["balance"]
    assert "error" in tl.get_customer("C9999")
    assert "error" in tl.find_customer("Nobody Here")


def test_splits_are_disjoint_by_customer_and_deterministic():
    train, ev = tl.questions("train"), tl.questions("eval")
    assert len(ev) == tl.N_EVAL_CUSTOMERS * 2
    assert len(train) == (tl.N_CUSTOMERS - tl.N_EVAL_CUSTOMERS) * 2
    assert not {r["customer_id"] for r in train} & {r["customer_id"] for r in ev}
    assert tl.questions("train") == train
    assert {r["level"] for r in train} == {1, 2}
    assert all(r["level"] == 1 for r in tl.questions("eval", l2_share=0.0))


def _conv(calls, results, final):
    conv = [{"role": "user", "content": "q"}]
    for call, result in zip(calls, results, strict=True):
        conv.append({"role": "assistant", "content": "", "tool_calls": [call]})
        conv.append({"role": "tool", "name": "x", "content": str(result)})
    conv.append({"role": "assistant", "content": final})
    return conv


def test_score_correct_rollout_both_call_shapes():
    c = tl.customers()[0]
    calls = [
        {"id": "a", "function": {"name": "find_customer", "arguments": f'{{"name": "{c["name"]}"}}'}},
        {"name": "get_customer", "parameters": {"customer_id": c["customer_id"]}},
    ]
    results = [tl.find_customer(c["name"]), tl.get_customer(c["customer_id"])]
    s = tl.score(_conv(calls, results, f"The balance is *{c['balance']}*."), "", c["balance"], "balance")
    assert s["n_tool_calls"] == 2
    assert s["tool_ok"] == 1.0 and s["hallucinated_ids"] == 0
    # the final text is passed separately; score it on the real reply
    final = f"The balance is *{c['balance']}*."
    s = tl.score(_conv(calls, results, final), final, c["balance"], "balance", c["customer_id"])
    assert (s["correct"], s["format"], s["tool_ok"]) == (1.0, 1.0, 1.0)


def test_real_record_of_the_wrong_customer_is_not_tool_ok():
    # baseline 5 Nov: Tiny Aya copied an example ID and answered from someone else's record
    asked, other = tl.customers()[0], tl.customers()[23]
    calls = [{"name": "get_customer", "arguments": {"customer_id": other["customer_id"]}}]
    final = f"*{other['balance']}*"
    s = tl.score(_conv(calls, [tl.get_customer(other["customer_id"])], final), final,
                 asked["balance"], "balance", asked["customer_id"])
    assert s["tool_ok"] == 0.0 and s["wrong_customer"] == 1 and s["hallucinated_ids"] == 0


def test_tool_docstrings_have_no_real_ids_or_names():
    import inspect
    docs = " ".join(inspect.getdoc(f) for f in tl.TOOLS)
    assert not any(c["customer_id"] in docs or c["name"] in docs for c in tl.customers())


def test_score_hallucinated_id_and_guess():
    c = tl.customers()[0]
    calls = [{"name": "get_customer", "arguments": {"customer_id": "C4242"}}]
    final = "*Premium*"
    s = tl.score(_conv(calls, [tl.get_customer("C4242")], final), final, c["plan"], "plan")
    assert s["hallucinated_ids"] == 1 and s["tool_ok"] == 0.0
    s = tl.score([], "**bold** but no answer", c["plan"], "plan")
    assert s == {**s, "correct": 0.0, "tool_ok": 0.0, "format": 0.0, "n_tool_calls": 0}


def test_folded_tool_error_breaks_tool_ok():
    # AgentTune folds tool results into user turns for Tiny Aya
    c = tl.customers()[0]
    conv = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"name": "find_customer",
                                                            "arguments": {"name": "Nobody"}}]},
        {"role": "user", "content": tl.TOOL_RESULT_FORMAT.format(
            name="find_customer", content=str(tl.find_customer("Nobody")))},
        {"role": "assistant", "content": "*Basic*"},
    ]
    s = tl.score(conv, "*Basic*", c["plan"], "plan")
    assert s["tool_ok"] == 0.0 and s["hallucinated_ids"] == 0


def test_harness_prompt_formats_like_agenttune():
    # AgentTune fills {tools} with str.format, so literal braces must be doubled
    text = tl.TOOLS_FALLBACK_PROMPT.format(tools="[]")
    assert '[{"tool_name": "<name>", "parameters": {"<argument>": <value>}}]' in text


def test_numeric_tolerance_and_text_match():
    assert tl._matches("£42.10", "42.10", "balance")
    assert not tl._matches("42.2", "42.10", "balance")
    assert tl._matches("12.3 GB", "12.3", "data_left_gb")
    assert tl._matches("premium.", "Premium", "plan")
    assert tl.starred("*45.1\n*") == "45.1"
    assert tl.score([], "*45.1\n*", "45.1", "data_left_gb")["format"] == 1.0


def test_reward_fn_logs_rows_and_debug(tmp_path):
    c = tl.customers()[0]
    final = f"*{c['plan']}*"
    conv = _conv([{"name": "get_customer", "arguments": {"customer_id": c["customer_id"]}}],
                 [tl.get_customer(c["customer_id"])], final)
    fn = tl.make_reward_fn(log_path=tmp_path / "rewards.jsonl")
    r = fn(["x", "y"], answer=[c["plan"]] * 2, field=["plan"] * 2, qid=["q1", "q2"], level=[1, 1],
           customer_id=[c["customer_id"]] * 2, conversations=[conv, []], responses=[final, "no idea"])
    assert r == pytest.approx([1.7, 0.0])
    assert len((tmp_path / "rewards.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "reward_inputs_debug.json").exists()
