"""SEL-91: meta-healing (reflexive.meta_heal). Every model is a fake: these tests make no API call."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reflexive import meta_heal as mh
from reflexive.actions import ActionFinding
from reflexive.monitor import BLOCK_MESSAGE, ActionGuard

SECRET = "EXPECTED-ACTION-SENTINEL"


def call(name="cancel_pending_order", **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def rec(domain="retail", task_id="1", passed=False, sample=0, words="Cancel order #W1 please."):
    """A saved conversation, with an answer key a rule must never see."""
    return {
        "domain": domain,
        "task_id": task_id,
        "sample": sample,
        "outcome": {
            "passed": passed,
            "reward": float(passed),
            "checks": {"db": float(passed), "note": SECRET},
        },
        "task": {"expected_actions": [SECRET], "instructions": SECRET},
        "agent_view": [
            {"role": "system", "content": "POLICY: refunds go to the original payment method."},
            {"role": "user", "content": words},
            {"role": "assistant", "content": "Cancelling #W1 now.", "tool_calls": [call(order_id="#W1")]},
            {"role": "tool", "content": '{"status": "cancelled"}'},
            {"role": "assistant", "content": "Done: your refund goes to your gift card."},
        ],
    }


class Fake:
    """A fake model: records every request, answers with ``answer(messages)``."""

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def __call__(self, messages):
        self.seen.append(messages)
        return self.answer(messages)


REFUND = mh.Candidate(
    "wrong_refund_method",
    "The agent refunds to a method the policy doesn't allow.",
    "call",
    "Does the change refund to a payment method other than the original one?",
)
SAID = mh.Candidate(
    "wrong_closing_statement",
    "The agent tells the customer something the tool results contradict.",
    "conversation",
    "Does the agent tell the customer something false?",
)


def violated(evidence="gift card"):
    return Fake(lambda m: {"violated": True, "evidence": evidence, "reason": "r"})


# --- What a rule reads ------------------------------------------------------------------------


def test_a_rule_never_reads_the_answer_key_or_the_outcome():
    for c in (REFUND, SAID):
        f = Fake(lambda m: {"violated": False})
        mh.candidate_findings(c, rec(), f)
        assert f.seen, c.name
        for messages in f.seen:
            text = json.dumps(messages)
            assert SECRET not in text and "expected_actions" not in text and '"passed"' not in text


def test_a_call_rule_sees_the_conversation_before_the_call_and_the_call_itself():
    f = Fake(lambda m: {"violated": False})
    mh.candidate_findings(REFUND, rec(), f)
    (messages,) = f.seen
    user = messages[1]["content"]
    assert "POLICY: refunds" in user and "PROPOSED CHANGE" in user and "cancel_pending_order" in user
    assert "Cancel order #W1 please." in user and "gift card" not in user  # the later turns aren't shown


def test_a_conversation_rule_sees_everything_including_tool_results_and_is_never_live():
    f = violated("your refund goes to your gift card")
    (hit,) = mh.candidate_findings(SAID, rec(), f)
    assert hit["flagged"] and '"status": "cancelled"' in f.seen[0][1]["content"]
    assert mh.Candidate.from_dict({**SAID.to_dict(), "live": True, "block_message": "x"}).live is False


def test_made_up_evidence_and_errors_never_flag():
    assert mh.candidate_findings(SAID, rec(), violated("the agent swore at me")) == []

    def boom(m):
        raise TimeoutError("slow")

    (e,) = mh.candidate_findings(SAID, rec(), Fake(boom))
    assert not e["flagged"] and "TimeoutError" in e["error"]


def test_evidence_must_come_from_the_conversation_not_the_rule_or_the_policy():
    c = mh.Candidate(
        "x", "The agent skips a step.", "conversation", "Does the CONVERSATION show a skipped step?"
    )
    assert mh.candidate_findings(c, rec(), violated("show a skipped step")) == []  # the rule's own words
    assert mh.candidate_findings(c, rec(), violated("the original payment method")) == []  # the policy's
    (hit,) = mh.candidate_findings(c, rec(), violated("your refund goes to your gift card"))
    assert hit["flagged"]


def test_only_state_changing_calls_are_checked():
    r = rec()
    r["agent_view"][2]["tool_calls"] = [call("get_order_details", order_id="#W1")]
    f = violated()
    assert mh.candidate_findings(REFUND, r, f) == [] and f.seen == []


# --- Candidates from the proposer -------------------------------------------------------------


def test_candidates_are_cleaned_and_live_needs_a_block_message():
    c = mh.Candidate.from_dict({"name": "Wrong Item!", "applies_to": "call", "question": "q?", "live": True})
    assert c.name == "wrong_item" and c.live is False  # no block message: after the fact
    c = mh.Candidate.from_dict(
        {"name": "x", "question": "q?", "live": True, "block_message": "Check the item."}
    )
    assert c.live and c.block_message == "Check the item."
    with pytest.raises(ValueError):
        mh.Candidate.from_dict({"name": "x", "question": "q", "applies_to": "turn"})


def test_the_proposer_sees_unexplained_failures_and_passes_but_never_the_held_out_third(monkeypatch):
    monkeypatch.setattr(mh, "explained", lambda r: r["task_id"] == "2")
    recs = [rec(task_id="1"), rec(task_id="2"), rec(task_id="3", passed=True)]
    answer = {
        "rules": [
            REFUND.to_dict(),
            REFUND.to_dict(),
            {"name": "bad"},
            *[{"name": f"r{i}", "question": "q"} for i in range(12)],
        ]
    }
    f = Fake(lambda m: answer)
    cands, meta = mh.propose(recs, f, n_passes=5)
    assert meta["unexplained_failures"] == ["retail:1#0"] and meta["passes_shown"] == ["retail:3#0"]
    assert (
        len(cands) == mh.MAX_CANDIDATES and cands[0].name == "wrong_refund_method"
    )  # deduped, bad one dropped
    user = f.seen[0][1]["content"]
    assert "retail:1#0: FAILED (failed checks: db)" in user and "retail:2#0" not in user
    assert "RETAIL POLICY" in user


def test_the_split_is_by_task_stratified_and_fixed_by_the_seed():
    recs = [
        rec(d, str(i), passed=i % 2 == 0, sample=s)
        for d in ("retail", "airline")
        for i in range(30)
        for s in (0, 1)
    ]
    prop, held = mh.split(recs, seed=0)
    assert len(held) == 40 and len(prop) == 80
    tasks = lambda rs: {mh.task_key(r) for r in rs}  # noqa: E731
    assert not tasks(prop) & tasks(held)  # both samples of a task on one side
    for d in ("retail", "airline"):
        for p in (True, False):
            assert sum(r["domain"] == d and mh.passed(r) is p for r in held) == 10
    assert mh.split(recs, seed=0) == (prop, held) and mh.split(recs, seed=1)[1] != held


# --- The bar ----------------------------------------------------------------------------------


def test_the_bar_is_stage2s(monkeypatch):
    monkeypatch.setattr(mh, "explained", lambda r: r["task_id"].startswith("x"))
    held = (
        [rec(task_id=f"u{i}") for i in range(10)]
        + [rec(task_id="x1")]
        + [rec(task_id=f"p{i}", passed=True) for i in range(10)]
    )

    which = {}  # the fake checker flags the conversation being scored if its task is in the set

    def checker(m):
        return {"violated": which["hit"], "evidence": "#W1", "reason": "r"}

    scores = []
    for name, hit_ids in {
        "two_of_ten": {"u0", "u1"},
        "one_of_ten": {"u0"},
        "two_passes": {"u0", "u1", "p0", "p1"},
    }.items():
        s = mh.Score(name)
        c = mh.Candidate(name, "w", "call", "q?")
        for r in held:
            which["hit"] = r["task_id"] in hit_ids
            s_one = mh.score([c], [r], checker)[0]
            for k in ("unexplained_failures", "other_failures", "passes"):
                getattr(s, k)[0] += getattr(s_one, k)[0]
                getattr(s, k)[1] += getattr(s_one, k)[1]
        scores.append(s)
    two, one, passes = scores
    assert two.unexplained_failures == [2, 10] and two.other_failures == [0, 1] and two.passes == [0, 10]
    assert two.meets_bar and not one.meets_bar and not passes.meets_bar
    assert mh.shortlist(scores) == ["two_of_ten"]
    errs = mh.Score("e", unexplained_failures=[5, 10], errors=1)
    assert not errs.meets_bar  # a checker error means the counts can't be trusted


def test_at_most_three_are_shortlisted():
    scores = [mh.Score(f"r{i}", unexplained_failures=[2 + i, 10]) for i in range(5)]
    assert mh.shortlist(scores) == ["r4", "r3", "r2"]


def test_a_judge_failure_while_sorting_explained_from_unexplained_stops_the_run(monkeypatch):
    import reflexive.hallucination as hallucination
    import reflexive.intent as intent

    none = SimpleNamespace(failure=hallucination.FailureType.NONE)
    monkeypatch.setattr(hallucination, "classify_record", lambda r: none)
    monkeypatch.setattr(
        intent,
        "intent_findings",
        lambda r, rules=None: [{"rule": "no_yes_judge_error", "live": False, "reason": "no key"}],
    )
    with pytest.raises(RuntimeError, match="no key"):
        mh.explained(rec(words="hello"))


# --- In the checker and the guard -------------------------------------------------------------


def test_the_rules_file_can_be_set_and_a_missing_one_is_reported(tmp_path, monkeypatch):
    p = tmp_path / "meta_rules.json"
    p.write_text(json.dumps({"accepted": [REFUND.to_dict()]}))
    monkeypatch.setenv("REFLEXIVE_META_RULES", str(p))
    assert mh.rules_path() == p and [c.name for c in mh.accepted()] == ["wrong_refund_method"]
    monkeypatch.setenv("REFLEXIVE_META_RULES", str(tmp_path / "missing.json"))
    with pytest.warns(UserWarning, match="no meta-healing rules loaded"):
        assert mh.accepted() == []


def test_accepted_rules_load_into_rules_and_judges_off_means_they_never_fire(tmp_path):
    p = tmp_path / "meta_rules.json"
    assert mh.meta_rules(p) == []  # nothing accepted yet: RULES is just "no yes"
    live = {**REFUND.to_dict(), "live": True, "block_message": "Refunds go to the original payment method."}
    p.write_text(json.dumps({"accepted": [live, SAID.to_dict()]}))
    (rule,) = mh.meta_rules(p)  # the conversation rule is after the fact: not a call rule
    assert rule.__name__ == "meta_wrong_refund_method"  # ty: ignore[unresolved-attribute]
    assert rule([], call(order_id="#W1"), "policy") is None  # conftest switches the judges off


def test_a_live_meta_rule_blocks_with_its_own_message_and_errors_never_block():
    c = mh.Candidate.from_dict(
        {**REFUND.to_dict(), "live": True, "block_message": "Use the original payment method."}
    )
    view = rec()["agent_view"][:2]
    f = mh.as_rule(c, llm=violated("#W1"))(view, call(order_id="#W1"), "policy")
    assert isinstance(f, ActionFinding) and (f.type, f.rule, f.live) == ("META", "wrong_refund_method", True)
    assert f.message == "Note from the system, not the customer: Use the original payment method."

    g = ActionGuard("act", frozenset({"cancel_pending_order"}), "policy", mh.as_rule(c, llm=violated("#W1")))
    reply = {"role": "assistant", "content": "", "tool_calls": [call(order_id="#W1")]}
    assert g.check(view, reply, agent_turn=2) is True
    assert g.reasked(view, 2)[-1]["content"] == f.message and g.reasks[0]["text"] == f.message

    def boom(m):
        raise RuntimeError("down")

    e = mh.as_rule(c, llm=Fake(boom))(view, call(order_id="#W1"), "policy")
    assert e is not None and e.live is False and e.rule == "wrong_refund_method_error"


def test_the_no_yes_rule_still_re_asks_with_block_message():
    def no_yes(v, c, p):
        return ActionFinding("INTENT", "no_yes", "no yes yet", live=True)

    g = ActionGuard("act", frozenset({"cancel_pending_order"}), "policy", no_yes)
    view = rec()["agent_view"][:2]
    assert g.check(view, {"role": "assistant", "tool_calls": [call(order_id="#W1")]}, 2)
    assert g.reasked(view, 2)[-1]["content"] == BLOCK_MESSAGE


def test_the_llm_caches_by_request(tmp_path):
    n = []

    def completion(**kw):
        n.append(kw)
        assert kw["temperature"] == 0 and kw["model"] == mh.RULE_MODEL
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"violated": false}'))]
        )

    llm = mh.LLM(mh.RULE_MODEL, completion, tmp_path / "c.jsonl")
    m = [{"role": "user", "content": "x"}]
    assert llm(m) == llm(m) == {"violated": False} and len(n) == 1
    again = mh.LLM(mh.RULE_MODEL, completion, tmp_path / "c.jsonl")
    assert again(m) == {"violated": False} and len(n) == 1 and again.calls == 0


# --- Rounds -----------------------------------------------------------------------------------


def test_round_one_is_unchanged_and_later_rounds_add_guidance_and_what_was_tried():
    fails = [rec("retail", "1"), rec("retail", "2"), rec("airline", "3")]
    one = mh.proposer_request(fails, [])
    assert one[0]["content"] == mh.PROPOSER_SYSTEM and "ALREADY TRIED" not in one[1]["content"]
    two = mh.proposer_request(fails, [], round=2, tried=[SAID], limit=5)
    assert "ROUND 2" in two[0]["content"] and "airline 1, retail 2" in two[0]["content"]
    assert "at most 5 rules" in two[0]["content"]
    assert two[1]["content"].startswith("ALREADY TRIED:\n- wrong_closing_statement (conversation)")


def test_a_later_round_keeps_within_the_ten_and_drops_repeats(monkeypatch):
    monkeypatch.setattr(mh, "explained", lambda r: False)
    answer = {"rules": [SAID.to_dict(), *[{"name": f"r{i}", "question": "q"} for i in range(8)]]}
    cands, meta = mh.propose([rec()], Fake(lambda m: answer), round=2, tried=[SAID], limit=5)
    assert [c.name for c in cands] == ["r0", "r1", "r2", "r3", "r4"] and meta["round"] == 2


def test_the_script_shares_the_ten_across_rounds_and_shortlists_across_them(tmp_path, monkeypatch):
    from _scripts import load_script

    s = load_script("heal/meta_heal")
    monkeypatch.setattr(s, "OUT", tmp_path)
    monkeypatch.setattr(s, "METRICS", tmp_path / "m.json")
    base = {
        "seed": 0,
        "proposer_model": "p",
        "propose_ids": [],
        "holdout_ids": [],
        "unexplained_failures": [],
        "proposed": 5,
    }
    one = [{**SAID.to_dict(), "name": f"a{i}"} for i in range(5)]
    s.write(s.paths(1)[0], {**base, "candidates": one})
    good = mh.Score("a1", unexplained_failures=[2, 5]).to_dict()
    s.write(s.paths(1)[1], {"scores": [mh.Score("a0").to_dict(), good]})
    assert s.rounds() == [1] and s.shortlist() == ["a1"]
    s.write(s.paths(2)[0], {**base, "candidates": [{**SAID.to_dict(), "name": f"b{i}"} for i in range(5)]})
    a = SimpleNamespace(round=3, seed=0, passes=12)
    with pytest.raises(SystemExit, match="allows 10"):
        s.cmd_propose([rec()], a)
    s.write_metrics(None)
    m = json.loads((tmp_path / "m.json").read_text())
    assert [r["round"] for r in m["rounds"]] == [1, 2] and m["meets_bar"] == ["a1"]


def test_scoring_some_candidates_again_keeps_the_others_and_the_recorded_accept(tmp_path, monkeypatch):
    from _scripts import load_script

    s = load_script("heal/meta_heal")
    monkeypatch.setattr(s, "OUT", tmp_path)
    monkeypatch.setattr(s, "METRICS", tmp_path / "m.json")
    rules = tmp_path / "meta_rules.json"
    rules.write_text(json.dumps({"names": [], "reviewer": "Charlie", "date": "2026-11-21", "accepted": []}))
    monkeypatch.setenv("REFLEXIVE_META_RULES", str(rules))
    recs = [rec(task_id=str(i)) for i in range(6)]
    _, held = mh.split(recs)
    cands = [{**SAID.to_dict(), "name": n} for n in ("a0", "a1")]
    base = {"seed": 0, "proposer_model": "p", "propose_ids": [], "unexplained_failures": [], "proposed": 2}
    s.write(s.paths(1)[0], {**base, "holdout_ids": [mh.example_id(r) for r in held], "candidates": cands})
    good = mh.Score("a1", unexplained_failures=[2, 5]).to_dict()
    s.write(s.paths(1)[1], {"scores": [mh.Score("a0").to_dict(), good]})
    monkeypatch.setattr(mh, "score", lambda cs, held, llm: [mh.Score(c.name) for c in cs])
    monkeypatch.setattr(mh, "LLM", lambda *a, **k: SimpleNamespace(calls=0))
    assert s.cmd_score(recs, SimpleNamespace(round=1, only="a0")) == 0
    kept = json.loads(s.paths(1)[1].read_text())["scores"]
    assert [x["name"] for x in kept] == ["a0", "a1"] and kept[1]["meets_bar"]
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["meets_bar"] == ["a1"] and m["accepted"] == {
        "names": [],
        "reviewer": "Charlie",
        "date": "2026-11-21",
    }
