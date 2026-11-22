"""Meta-healing v2 (reflexive.meta_heal_v2). Every model is a fake: no API calls."""

from __future__ import annotations

import json

import pytest

from reflexive import meta_heal as mh
from reflexive import meta_heal_v2 as v2

KEY = "ANSWER-KEY-SENTINEL"


def call(name="cancel_pending_order", **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def rec(task_id="1", passed=False, sample=0, domain="retail"):
    long_result = '{"items": [' + ", ".join(f'{{"item_id": "{i:010d}"}}' for i in range(90)) + "]}"
    return {
        "domain": domain,
        "task_id": task_id,
        "sample": sample,
        "outcome": {"passed": passed, "checks": {"db": float(passed)}},
        "agent_view": [
            {"role": "system", "content": "POLICY TEXT"},
            {"role": "user", "content": "Cancel order #W1."},
            {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", order_id="#W1")]},
            {"role": "tool", "content": long_result},
            {"role": "assistant", "content": "Cancelling.", "tool_calls": [call(order_id="#W1", reason="x")]},
        ],
    }


def expected(r):
    return [f'cancel_pending_order({{"order_id": "{KEY}"}})']


class Fake:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def __call__(self, messages):
        self.seen.append(messages)
        return self.answer(messages) if callable(self.answer) else self.answer


RULE = {"name": "r1", "what": "w", "applies_to": "call", "tools": ["cancel_pending_order"], "question": "q?"}


def test_the_proposer_sees_the_answer_key_and_full_tool_results_and_no_error_kind_is_named():
    msgs = v2.proposer_request([rec()], [rec("2", passed=True)], expected)
    user = msgs[1]["content"]
    assert KEY in user and "EXPECTED ACTIONS" in user and "0000000089" in user  # past v1's 600 characters
    assert user.count("EXPECTED ACTIONS") == 1  # only failures carry the key
    assert (
        "mismatch" not in msgs[0]["content"].lower() and "match what the customer" not in msgs[0]["content"]
    )


def test_rules_are_checked_by_full_gpt41_and_never_see_the_answer_key():
    (c,) = v2.parse({"rules": [RULE]})
    assert c.model == "gpt-4.1-2025-04-14" and c.tool_chars == 6000 and c.tools == ("cancel_pending_order",)
    chk = Fake({"violated": False})
    v2.findings_by_record([c], [rec()], chk, workers=1)
    assert chk.seen and all(KEY not in json.dumps(m) and "EXPECTED" not in json.dumps(m) for m in chk.seen)


def test_learning_rounds_feed_back_what_each_rule_caught_and_wrongly_flagged():
    recs = [rec("1"), rec("2"), rec("3", passed=True), rec("4", passed=True)]
    rounds = iter(
        [{"rules": [RULE]}, {"rules": [{**RULE, "name": "r2"}]}, {"rules": [{**RULE, "name": "r3"}]}]
    )
    proposer = Fake(lambda m: next(rounds))
    chk = Fake({"violated": True, "evidence": "Cancel order #W1.", "reason": "because"})
    out = v2.learn(
        recs,
        proposer,
        chk,
        rounds=3,
        n_passes_scored=2,
        expected=expected,
        explained=lambda r: r["task_id"] == "2",
        log=lambda s: None,
    )
    assert out["unexplained_failures"] == ["retail:1#0"]
    assert [r["round"] for r in out["rounds"]] == [1, 2, 3] and out["final"][0]["name"] == "r3"
    assert out["rounds"][0]["counts"]["r1"] == {"failures": [1, 1], "passes": [2, 2]}
    second = proposer.seen[1]
    assert second[-2]["role"] == "assistant" and "r1: caught 1/1 failures" in second[-1]["content"]
    assert "wrongly flagged 2/2 passes" in second[-1]["content"] and "because" in second[-1]["content"]
    assert len(proposer.seen) == 3


def test_the_held_out_bar_counts_added_catches_passes_at_worst_and_rediscovery():
    (c,) = v2.parse({"rules": [RULE]})
    recs = [rec(str(i)) for i in range(10)] + [rec(f"p{i}", passed=True) for i in range(10)]
    frozen = {
        mh.example_id(r): {"tal": False, "no_yes": False, "mismatch": r["task_id"] in {"0", "1"}}
        for r in recs
    }

    def answer(m):
        text = m[1]["content"]
        return (
            {"violated": True, "evidence": "Cancel order #W1.", "reason": "r"}
            if "flag-me" in text
            else {"violated": False}
        )

    for r in recs[2:5] + recs[10:11]:  # 3 added catches, 1 pass
        r["agent_view"][1]["content"] = "Cancel order #W1. flag-me"
    (s,) = v2.held_out([c], recs, frozen, Fake(answer), workers=1)
    assert s.added == [3, 8] and s.failures == [3, 10] and s.passes == [1, 10] and s.rediscovers == [0, 2]
    assert s.meets_bar and v2.shortlist([s]) == ["r1"]
    s.passes_errored = 1
    assert not s.meets_bar  # 2/10 passes at worst


def test_the_script_allows_at_most_three_rounds():
    from _scripts import load_script

    with pytest.raises(SystemExit, match="at most 3"):
        load_script("heal/meta_heal_v2").main(["learn", "--rounds", "4"])


def test_the_layers_table_is_cumulative_and_counts_failures_and_passes():
    ids = {"f1": False, "f2": False, "f3": False, "f4": False, "p1": True, "p2": True}
    frozen = {
        "f1": {"tal": True, "no_yes": False, "mismatch": False},
        "f2": {"tal": False, "no_yes": True, "mismatch": True},
        "f3": {"tal": False, "no_yes": False, "mismatch": True},
        "f4": {"tal": False, "no_yes": False, "mismatch": False},
        "p1": {"tal": False, "no_yes": False, "mismatch": True},
        "p2": {"tal": False, "no_yes": False, "mismatch": False},
    }
    scores = [
        {"name": "r1", "flagged": [{"id": "f4"}, {"id": "f3"}]},
        {"name": "r2", "flagged": [{"id": "p2"}]},
    ]
    rows = v2.layers(ids, frozen, scores, ["r1"])
    assert [r["failures"] for r in rows] == [[1, 4], [2, 4], [3, 4], [4, 4]]
    assert [r["passes"] for r in rows] == [[0, 2], [0, 2], [1, 2], [1, 2]]  # r2 isn't accepted
    assert len(v2.layers(ids, frozen, scores, [])) == 3  # nothing accepted: no fourth layer


def test_a_round_with_no_usable_rules_keeps_the_last_rounds_rules():
    recs = [rec("1"), rec("3", passed=True)]
    rounds = iter([{"rules": [RULE]}, {"rules": [{"name": "bad"}]}])  # round 2: no question, unusable
    out = v2.learn(
        recs,
        Fake(lambda m: next(rounds)),
        Fake({"violated": False}),
        rounds=3,
        expected=expected,
        explained=lambda r: False,
        log=lambda s: None,
    )
    assert [r["round"] for r in out["rounds"]] == [1] and [c["name"] for c in out["final"]] == ["r1"]
    assert out["passes_overlap"] == ["retail:3#0"]  # one pass: shown and scored both


def test_a_rules_domains_come_from_its_name_or_its_tools():
    def cand(name, tools=()):
        return mh.Candidate(name, "w", "conversation" if not tools else "call", "q?", tools=tuple(tools))

    assert v2.with_domains(cand("retail_transfer_to_human")).domains == ("retail",)
    assert v2.with_domains(cand("baggage_check", ["update_reservation_baggages"])).domains == ("airline",)
    assert v2.with_domains(cand("generic_rule")).domains == ()
    (c,) = v2.parse({"rules": [{**RULE, "name": "telecom_x", "tools": []}]})
    assert c.domains == () and v2.with_domains(c).domains == ("telecom",)  # learning is left as it ran


def test_a_rule_is_scored_and_run_only_on_its_own_domains():
    rule = {**RULE, "name": "retail_r", "applies_to": "conversation", "tools": []}
    (c,) = v2.parse({"rules": [rule]})
    flag = Fake({"violated": True, "evidence": "Cancel order #W1.", "reason": "r"})
    recs = [rec("1"), rec("p1", passed=True), rec("a1", passed=True, domain="airline")]
    frozen = {mh.example_id(r): {"tal": False, "no_yes": False, "mismatch": False} for r in recs}
    (s,) = v2.held_out([c], recs, frozen, flag, workers=1)
    assert s.passes == [1, 1] and s.failures == [1, 1]  # the airline pass isn't counted
    assert all("airline" not in json.dumps(m) for m in flag.seen) and len(flag.seen) == 2
    c = v2.with_domains(c)
    assert mh.conversation_findings(rec("a1", domain="airline"), path=_rules_file([c]), llm=flag) == []
    airline = v2.with_domains(mh.Candidate.from_dict({**RULE, "name": "airline_x", "tools": []}))
    assert not airline.checks("cancel_pending_order") and airline.checks("cancel_reservation")


def _rules_file(cands):
    import tempfile
    from pathlib import Path

    p = Path(tempfile.mkdtemp()) / "meta_rules.json"
    p.write_text(json.dumps({"accepted": [c.to_dict() for c in cands]}))
    return p


def _script(tmp_path, monkeypatch):
    from _scripts import load_script

    s = load_script("heal/meta_heal_v2")
    monkeypatch.setattr(s, "OUT", tmp_path)
    monkeypatch.setattr(s, "METRICS", tmp_path / "m.json")
    monkeypatch.setenv("REFLEXIVE_META_RULES", str(tmp_path / "meta_rules.json"))
    (c,) = v2.parse({"rules": [{**RULE, "name": "retail_r"}]})
    learned = {
        "unexplained_failures": [],
        "passes_shown": [],
        "passes_scored": [],
        "rounds": [],
        "final": [c.to_dict()],
    }
    s.write(tmp_path / "learn.json", learned)
    score = {**v2.HeldOut("retail_r", added=[3, 5]).to_dict(), "flagged": []}
    s.write(
        tmp_path / "scores.json",
        {"shortlist": ["retail_r"], "scores": [score], "frozen_rates": {}, "frozen": {}, "ids": {}},
    )
    return s


def test_accept_keeps_v1s_accepted_rules_and_a_rescore_keeps_the_accept(tmp_path, monkeypatch):
    from types import SimpleNamespace

    s = _script(tmp_path, monkeypatch)
    v1_rule = {"name": "v1_rule", "what": "w", "applies_to": "call", "question": "q?"}
    (tmp_path / "meta_rules.json").write_text(json.dumps({"ticket": "SEL-91", "accepted": [v1_rule]}))
    assert s.cmd_accept(SimpleNamespace(names="retail_r", reviewer="Charlie")) == 0
    cfg = json.loads((tmp_path / "meta_rules.json").read_text())
    assert cfg["v1_accepted"] == [v1_rule] and [r["name"] for r in cfg["accepted"]] == ["retail_r"]
    assert cfg["accepted"][0]["domains"] == ["retail"]
    assert s.recorded_acceptance()["names"] == ["retail_r"]
    assert s.cmd_accept(SimpleNamespace(names="", reviewer=None)) == 0  # again, from a v2 file
    assert json.loads((tmp_path / "meta_rules.json").read_text())["v1_accepted"] == [v1_rule]


def test_review_of_an_unknown_rule_says_so(tmp_path, monkeypatch):
    from types import SimpleNamespace

    s = _script(tmp_path, monkeypatch)
    monkeypatch.setattr(s, "load", lambda episodes, split: [])
    with pytest.raises(SystemExit, match="no scored rule named 'retail_typo'"):
        s.cmd_review(SimpleNamespace(only="retail_typo", episodes=tmp_path))


def test_new_calls_are_counted_across_threads():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    def completion(**kw):
        msg = SimpleNamespace(content='{"violated": false}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    llm = mh.LLM("m", completion=completion)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: llm([{"role": "user", "content": str(i)}]), range(400)))
    assert llm.calls == 400
