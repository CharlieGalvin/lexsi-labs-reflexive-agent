import inspect
import random
from collections import Counter

import pytest

from reflexive import messy_lookup as ml


def test_database_is_fixed_and_names_unique():
    db = ml.customers()
    assert len(db) == ml.N_CUSTOMERS
    assert len({c["name"] for c in db}) == ml.N_CUSTOMERS
    assert db[0] == ml.customers.__wrapped__()[0]  # same seed, same data


def test_every_question_type_appears_and_splits_are_disjoint():
    train, ev = ml.questions("train"), ml.questions("eval")
    assert len(train) == (ml.N_CUSTOMERS - ml.N_EVAL_CUSTOMERS) * 3
    assert len(ev) == ml.N_EVAL_CUSTOMERS * 3
    assert set(Counter(r["level"] for r in train)) == set(ml.STYLES)
    assert not {r["customer_id"] for r in train} & {r["customer_id"] for r in ev}
    assert ml.questions("train") == train


def test_distractor_ids_stay_in_the_split():
    ev_ids = ml.eval_customer_ids()
    for r in ml.questions("train"):
        if r["level"] == "distractor":
            ids = [w.strip(".,?:") for w in r["question"].split() if w.startswith("C1")]
            assert len(ids) == 2 and not set(ids) & ev_ids


def test_messy_ids_are_rejected_by_the_tool():
    cid = ml.customers()[7]["customer_id"]
    assert "error" not in ml.get_customer(cid.lower())
    for messy in (f"c-{cid[1:]}", f"C {cid[1:]}", cid[1:]):
        assert "error" in ml.get_customer(messy)


def test_question_wording_per_type():
    q = {s: ml._question(s, "plan", "C1023", "C1088", random.Random(0)) for s in ml.STYLES}
    assert "c-1023" in q["lower_dash"] and "C 1023" in q["spaced"]
    assert "account number 1023" in q["number_only"]
    assert "C one zero two three" in q["spelled"]
    assert "C1023" in q["distractor"] and "C1088" in q["distractor"]


def test_style_weights_and_unknown_types():
    assert {r["level"] for r in ml.questions("eval", styles={"clean": 1})} == {"clean"}
    with pytest.raises(ValueError):
        ml.questions("eval", styles={"shouty": 1})


def test_tool_docstring_has_no_real_ids():
    doc = inspect.getdoc(ml.get_customer)
    assert not any(c["customer_id"] in doc for c in ml.customers())


def test_harness_prompt_formats_like_agenttune():
    # AgentTune fills {tools} with str.format, so literal braces must be doubled
    text = ml.TOOLS_FALLBACK_PROMPT.format(tools="[]")
    assert '[{"tool_name": "<name>", "parameters": {"<argument>": <value>}}]' in text


# --- scoring ---


def _conv(calls, results, final, folded=False):
    conv = [{"role": "user", "content": "q"}]
    for call, result in zip(calls, results, strict=True):
        conv.append({"role": "assistant", "content": "", "tool_calls": [call]})
        if folded:  # how AgentTune shows tool results to Tiny Aya
            conv.append({"role": "user", "content": ml.TOOL_RESULT_FORMAT.format(
                name="get_customer", content=str(result))})
        else:
            conv.append({"role": "tool", "name": "get_customer", "content": str(result)})
    conv.append({"role": "assistant", "content": final})
    return conv


def test_correct_rollout_in_both_call_shapes():
    c = ml.customers()[0]
    final = f"The balance is *{c['balance']}*."
    for call in ({"id": "a", "function": {"name": "get_customer",
                                          "arguments": f'{{"customer_id": "{c["customer_id"]}"}}'}},
                 {"name": "get_customer", "parameters": {"customer_id": c["customer_id"]}}):
        conv = _conv([call], [ml.get_customer(c["customer_id"])], final)
        s = ml.score(conv, final, c["balance"], "balance", c["customer_id"])
        assert (s["correct"], s["tool_ok"], s["format"], s["n_tool_calls"]) == (1.0, 1.0, 1.0, 1)


def test_badly_normalised_id_then_retry():
    c = ml.customers()[0]
    messy = f"c-{c['customer_id'][1:]}"
    calls = [{"name": "get_customer", "arguments": {"customer_id": messy}},
             {"name": "get_customer", "arguments": {"customer_id": c["customer_id"]}}]
    results = [ml.get_customer(messy), ml.get_customer(c["customer_id"])]
    final = f"*{c['plan']}*"
    s = ml.score(_conv(calls, results, final, folded=True), final, c["plan"], "plan", c["customer_id"])
    # right answer in the end, but the first call failed: correct yes, tool_ok no
    assert s["correct"] == 1.0 and s["tool_ok"] == 0.0 and s["hallucinated_ids"] == 1


def test_real_record_of_the_wrong_customer_is_not_tool_ok():
    asked, other = ml.customers()[0], ml.customers()[23]
    calls = [{"name": "get_customer", "arguments": {"customer_id": other["customer_id"]}}]
    final = f"*{other['balance']}*"
    s = ml.score(_conv(calls, [ml.get_customer(other["customer_id"])], final), final,
                 asked["balance"], "balance", asked["customer_id"])
    assert s["tool_ok"] == 0.0 and s["wrong_customer"] == 1 and s["hallucinated_ids"] == 0


def test_no_tool_call_and_bold_only():
    c = ml.customers()[0]
    s = ml.score([], "**bold** but no answer", c["plan"], "plan", c["customer_id"])
    assert (s["correct"], s["tool_ok"], s["format"], s["n_tool_calls"]) == (0.0, 0.0, 0.0, 0)


def test_answer_matching():
    assert ml._matches("£42.10", "42.10", "balance")
    assert not ml._matches("42.2", "42.10", "balance")
    assert ml._matches("12.3 GB", "12.3", "data_left_gb")
    assert ml._matches("premium.", "Premium", "plan")
    assert ml.starred("*45.1\n*") == "45.1"


def test_reward_fn_logs_rows_and_debug(tmp_path):
    c = ml.customers()[0]
    final = f"*{c['plan']}*"
    conv = _conv([{"name": "get_customer", "arguments": {"customer_id": c["customer_id"]}}],
                 [ml.get_customer(c["customer_id"])], final)
    fn = ml.make_reward_fn(log_path=tmp_path / "rewards.jsonl")
    r = fn(["x", "y"], answer=[c["plan"]] * 2, field=["plan"] * 2, qid=["q1", "q2"],
           level=["clean"] * 2, customer_id=[c["customer_id"]] * 2,
           conversations=[conv, []], responses=[final, "no idea"])
    assert r == pytest.approx([1.7, 0.0])
    assert len((tmp_path / "rewards.jsonl").read_text().splitlines()) == 2
    assert (tmp_path / "reward_inputs_debug.json").exists()
