import inspect
from collections import Counter

import pytest

from reflexive import messy_lookup as ml
from reflexive import toy_lookup as tl


def test_every_question_type_appears_and_splits_are_disjoint():
    train, ev = ml.questions("train"), ml.questions("eval")
    assert len(train) == (tl.N_CUSTOMERS - tl.N_EVAL_CUSTOMERS) * 3
    assert len(ev) == tl.N_EVAL_CUSTOMERS * 3
    assert set(Counter(r["level"] for r in train)) == set(ml.STYLES)
    assert not {r["customer_id"] for r in train} & {r["customer_id"] for r in ev}
    assert ml.questions("train") == train


def test_distractor_ids_stay_in_the_split():
    ev_ids = tl.eval_customer_ids()
    for r in ml.questions("train"):
        if r["level"] == "distractor":
            other = [w.strip(".,?:") for w in r["question"].split() if w.startswith("C1")]
            assert len(other) == 2 and not set(other) & ev_ids


def test_messy_ids_are_rejected_by_the_tool():
    cid = tl.customers()[7]["customer_id"]
    assert "error" not in ml.get_customer(cid.lower())
    for messy in (f"c-{cid[1:]}", f"C {cid[1:]}", cid[1:]):
        assert "error" in ml.get_customer(messy)


def test_question_wording_per_type():
    rng = __import__("random").Random(0)
    q = {s: ml._question(s, "plan", "C1023", "C1088", rng) for s in ml.STYLES}
    assert "c-1023" in q["lower_dash"] and "C 1023" in q["spaced"]
    assert "account number 1023" in q["number_only"]
    assert "C one zero two three" in q["spelled"]
    assert "C1023" in q["distractor"] and "C1088" in q["distractor"]


def test_style_weights_and_unknown_types():
    only_clean = ml.questions("eval", styles={"clean": 1})
    assert {r["level"] for r in only_clean} == {"clean"}
    with pytest.raises(ValueError):
        ml.questions("eval", styles={"shouty": 1})


def test_tool_docstring_has_no_real_ids():
    doc = inspect.getdoc(ml.get_customer)
    assert not any(c["customer_id"] in doc for c in tl.customers())
    assert ml.get_customer.__name__ == "get_customer"  # scoring looks for this name
