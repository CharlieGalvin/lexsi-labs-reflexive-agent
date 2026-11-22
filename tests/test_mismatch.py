"""The hand-written "mismatch" rule (reflexive.mismatch) and its injected-error test. No API calls."""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest
from _scripts import load_script

from reflexive import meta_heal as mh
from reflexive.actions import OPTIONAL_RULES, RULES, ActionFinding, checker_with
from reflexive.intent import ALL_WRITE_TOOLS
from reflexive.mismatch import RULE, TOOLS, mismatch


def call(name, **args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


LONG = '{"order_id": "#W1", "items": [' + ", ".join(f'{{"item_id": "{i:010d}"}}' for i in range(150)) + "]}"
VIEW = [
    {"role": "system", "content": "POLICY"},
    {"role": "user", "content": "Swap the red one for the blue one in order #W1."},
    {"role": "assistant", "content": "", "tool_calls": [call("get_order_details", order_id="#W1")]},
    {"role": "tool", "content": LONG},
]


class Fake:
    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def __call__(self, messages):
        self.seen.append(messages)
        return self.answer


def test_it_is_opt_in_not_in_the_default_rules_and_is_live_with_its_own_message():
    assert mismatch not in RULES and OPTIONAL_RULES["mismatch"] is mismatch
    assert RULE.live and RULE.block_message and RULE.applies_to == "call"
    assert RULE.model == "gpt-4.1-2025-04-14"  # full GPT-4.1, pinned
    assert mh.default_rule_llm() is None  # conftest: judges off


def test_it_checks_only_the_listed_tools_and_never_fires_with_judges_off():
    yes = Fake({"violated": True, "evidence": "red one", "reason": "r"})
    rule = mh.as_rule(RULE, llm=yes, kind="INTENT", prefix="")
    assert rule(VIEW, call("modify_pending_order_address", order_id="#W1"), "p") is None and not yes.seen
    f = rule(
        VIEW, call("modify_pending_order_items", order_id="#W1", item_ids=["1"], new_item_ids=["2"]), "p"
    )
    assert f is not None and (f.type, f.rule, f.live) == ("INTENT", "mismatch", True)
    assert f.message.startswith("Note from the system, not the customer: this change may not match")
    assert mismatch(VIEW, call("cancel_pending_order", order_id="#W1"), "p") is None  # conftest: judges off
    assert set(TOOLS) <= ALL_WRITE_TOOLS


def test_the_checker_sees_long_tool_results_in_full():
    f = Fake({"violated": False})
    mh.as_rule(RULE, llm=f)(VIEW, call("cancel_pending_order", order_id="#W1"), "p")
    assert '"item_id": "0000000149"' in f.seen[0][1]["content"]  # past meta-healing's 1,200 characters


def test_corruptions_change_the_call_in_one_known_way():
    s = load_script("labels/rule_rates")
    rng = random.Random(0)
    seen = '"item_id": "1111111111" "item_id": "2222222222" "item_id": "3333333333" #W1234567 #W7654321'
    c, kind = s.corrupt(
        call(
            "exchange_delivered_order_items",
            order_id="#W1234567",
            item_ids=["1111111111"],
            new_item_ids=["2222222222"],
        ),
        seen,
        rng,
    )
    assert json.loads(c["function"]["arguments"])["new_item_ids"] == ["3333333333"] and "swapped" in kind
    c, kind = s.corrupt(call("cancel_pending_order", order_id="#W1234567", reason="x"), seen, rng)
    assert json.loads(c["function"]["arguments"])["order_id"] == "#W7654321"
    fl = [{"flight_number": "HAT1", "date": "2024-05-19"}, {"flight_number": "HAT2", "date": "2024-05-23"}]
    c, kind = s.corrupt(call("update_reservation_flights", reservation_id="ABC", flights=fl), "", rng)
    assert json.loads(c["function"]["arguments"])["flights"] == fl[:1] and kind == "a flight dropped"
    c, _ = s.corrupt(call("update_reservation_flights", reservation_id="ABC", flights=fl[:1]), "", rng)
    assert json.loads(c["function"]["arguments"])["flights"][0]["date"] == "2024-05-20"
    c, _ = s.corrupt(
        call("update_reservation_baggages", reservation_id="ABC", total_baggages=2, nonfree_baggages=0),
        "",
        rng,
    )
    assert json.loads(c["function"]["arguments"])["total_baggages"] == 3
    assert (
        s.corrupt(call("cancel_pending_order", order_id="#W1234567"), "#W1234567", rng) is None
    )  # no other order


def test_rule_rates_counts_a_checker_error_as_not_flagged_and_shows_the_worst_case(
    tmp_path, monkeypatch, capsys
):
    import reflexive.intent as intent
    from reflexive.splits import task_ids

    s = load_script("labels/rule_rates")
    t = task_ids("retail", "train")[0]
    rec = {
        "domain": "retail",
        "task_id": t,
        "outcome": {"passed": True},
        "messages": [],
        "agent_view": [{"role": "system", "content": "P"}],
    }
    (tmp_path / "S2-retail-trainpool.jsonl").write_text(json.dumps(rec) + "\n")

    def findings(r, rules=None):
        if rules and rules[0].__name__ == "mismatch":  # ty: ignore[unresolved-attribute]
            return [
                {
                    "turn": 1,
                    "tool": "x",
                    "live": False,
                    "rule": "mismatch_error",
                    "reason": "JSONDecodeError: runaway",
                }
            ]
        return []

    monkeypatch.setattr(intent, "intent_findings", findings)
    out = tmp_path / "o.json"
    assert s.main(["--episodes", str(tmp_path), "--json", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["rules"]["S2/mismatch"]["all"]["passes"] == {"flagged": 0, "n": 1, "errored_unflagged": 1}
    assert "at worst 1" in capsys.readouterr().out

    monkeypatch.setattr(
        intent, "intent_findings", lambda r, rules=None: [{"live": False, "reason": "Missing credentials"}]
    )
    assert s.main(["--episodes", str(tmp_path)]) == 1  # a setup error still stops the run


def test_an_arm_opts_in_with_checker_with(monkeypatch):
    import reflexive.actions as actions

    hit = ActionFinding("INTENT", "mismatch", "wrong item", live=True)
    monkeypatch.setitem(actions.OPTIONAL_RULES, "mismatch", lambda *a: hit)
    monkeypatch.setattr(actions, "RULES", [])
    assert checker_with()(VIEW, call("cancel_pending_order", order_id="#W1"), "p") is None
    assert checker_with("mismatch")(VIEW, call("cancel_pending_order", order_id="#W1"), "p") is hit
    with pytest.raises(ValueError, match="no optional rule"):
        checker_with("nope")


def test_validation_tasks_need_every_domains_file(tmp_path):
    s = load_script("labels/rule_rates")
    (tmp_path / "P-retail-val.jsonl").write_text('{"task_id": "7"}\n')
    assert s.val_tasks(tmp_path) is None  # airline and telecom missing: nothing would be dropped there
    for d in ("airline", "telecom"):
        (tmp_path / f"P-{d}-val.jsonl").write_text('{"task_id": "1"}\n')
    assert s.val_tasks(tmp_path) == {("retail", "7"), ("airline", "1"), ("telecom", "1")}


def test_a_catch_counts_only_when_the_original_is_not_flagged():
    s = load_script("labels/rule_rates")
    seen = '"item_id": "1111111111" "item_id": "2222222222"'
    calls = [call("return_delivered_order_items", order_id="#W1", item_ids=["1111111111"]) for _ in range(2)]
    recs = [
        {
            "_arm": "S2",
            "domain": "retail",
            "task_id": str(i),
            "outcome": {"passed": True},
            "agent_view": [
                {"role": "system", "content": "P"},
                {"role": "tool", "content": seen},
                {"role": "assistant", "content": "", "tool_calls": [c]},
            ],
        }
        for i, c in enumerate(calls)
    ]
    hit = ActionFinding("INTENT", "mismatch", "x", live=True)
    every = s.injected(recs, 2, lambda *a: hit)  # flags everything, originals included
    assert (every["tried"], every["caught"], every["caught_raw"], every["clean_flagged"]) == (2, 0, 2, 2)

    def only_bad(conversation, c, policy):
        return hit if len(json.loads(c["function"]["arguments"])["item_ids"]) == 2 else None

    good = s.injected(recs, 2, only_bad)
    assert (good["caught"], good["caught_raw"], good["clean_flagged"]) == (2, 2, 0)


def test_a_flat_call_can_be_corrupted():
    s = load_script("labels/rule_rates")
    flat = {"name": "cancel_pending_order", "arguments": json.dumps({"order_id": "#W1234567"})}
    c, _ = s.corrupt(flat, "#W1234567 #W7654321", random.Random(0))
    assert json.loads(c["arguments"])["order_id"] == "#W7654321"


def test_one_arm_with_two_splits_is_reported_per_split(tmp_path, monkeypatch):
    import reflexive.intent as intent
    from reflexive.splits import task_ids

    s = load_script("labels/rule_rates")
    t = task_ids("retail", "train")[0]
    rec = {"domain": "retail", "task_id": t, "outcome": {"passed": True}, "messages": [], "agent_view": []}
    for split in ("trainpool", "train"):
        (tmp_path / f"S2-retail-{split}.jsonl").write_text(json.dumps(rec) + "\n")
    for d in ("retail", "airline", "telecom"):
        (tmp_path / f"P-{d}-val.jsonl").write_text("")
    monkeypatch.setattr(intent, "intent_findings", lambda r, rules=None: [])
    out = tmp_path / "o.json"
    assert s.main(["--episodes", str(tmp_path), "--arms", "S2:trainpool,S2:train", "--json", str(out)]) == 0
    m = json.loads(out.read_text())["rules"]
    assert (
        m["S2:trainpool/no_yes"]["all"]["passes"]["n"] == 1
        and m["S2:train/no_yes"]["all"]["passes"]["n"] == 1
    )
    assert "S2/no_yes" not in m


def test_only_the_rule_checker_caps_its_answer():
    seen = []

    def completion(**kw):
        seen.append(kw)
        msg = SimpleNamespace(content='{"violated": false}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])

    mh.LLM("proposer", completion=completion)([{"role": "user", "content": "a"}])
    mh.LLM("rule", completion=completion, max_tokens=mh.MAX_ANSWER_TOKENS)([{"role": "user", "content": "b"}])
    assert "max_tokens" not in seen[0] and seen[1]["max_tokens"] == mh.MAX_ANSWER_TOKENS

    def cut(**kw):
        msg = SimpleNamespace(content='{"viol')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="length")])

    with pytest.raises(ValueError, match="cut off"):
        mh.LLM("rule", completion=cut, max_tokens=10)([{"role": "user", "content": "c"}])
