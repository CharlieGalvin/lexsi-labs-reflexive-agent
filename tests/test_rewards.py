"""Offline tests: no model, no GPU, no network. Run with `pytest`."""

from reflexive import trajectory as T
from reflexive.metrics import aggregate, episode_metrics
from reflexive.rewards import (
    build_reward_funcs,
    correctness_reward,
    groundedness_reward,
    make_efficiency_reward,
    make_validity_reward,
)
from reflexive.rollout import run_episode

Q = [{"role": "user", "content": "Does the gene X interact with Y?"}]


def call(sql, cid="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": cid, "type": "function",
         "function": {"name": "query_biogrid", "arguments": f'{{"sql_command": "{sql}"}}'}}]}


def result(content, cid="c1"):
    return {"role": "tool", "tool_call_id": cid, "content": content}


def answer(text):
    return {"role": "assistant", "content": text}


LEAN = [call("SELECT 1"), result("[[1]]"), answer("*Yes*")]
LOOPY = [call("SELECT 1"), result("[[1]]"), call("SELECT 1", "c2"), result("[[1]]", "c2"),
         call("SELECT 1", "c3"), result("[[1]]", "c3"), answer("*Yes*")]
NO_TOOLS_GUESS = [answer("*Yes*")]
ERRORY = [call("SELEC"), result('{"error": "syntax error"}'), call("SELECT 1", "c2"),
          result("[[1]]", "c2"), answer("*Yes*")]


def test_parse_counts():
    t = T.parse(LOOPY)
    assert t.n_calls == 3 and t.n_redundant == 2 and t.final_answer == "*Yes*"
    assert T.parse(ERRORY).n_errors == 1


def test_parse_cohere_fallback_format():
    turns = [answer('[{"tool_name": "query_biogrid", "parameters": {"sql_command": "SELECT 1"}}]'),
             {"role": "user", "content": "Tool result for query_biogrid: [[1]]\nUse this result..."},
             answer("*No*")]
    t = T.parse(turns)
    assert t.n_calls == 1 and t.results and t.final_answer == "*No*"


def test_efficiency_prefers_lean_and_blocks_guessing():
    eff = make_efficiency_reward(step_budget=3)
    lean, loopy, guess, errory = eff(None, [LEAN, LOOPY, NO_TOOLS_GUESS, ERRORY], answer=["Yes"] * 4)
    assert lean == 1.0
    assert loopy < lean and errory < lean
    assert guess == 0.0  # correct-but-ungrounded guess earns no efficiency credit


def test_efficiency_zero_when_wrong():
    assert make_efficiency_reward()(None, [LEAN], answer=["No"]) == [0.0]


def test_groundedness():
    unans_abstain = [call("SELECT 1"), result("[]"),
                     answer("The information is not available in the database.")]
    unans_confident = [call("SELECT 1"), result("[]"), answer("It was discovered in *1979*.")]
    s = groundedness_reward(None, [LEAN, NO_TOOLS_GUESS, unans_abstain, unans_confident],
                            answer=["Yes", "Yes", "UNANSWERABLE", "UNANSWERABLE"])
    assert s[0] > 0 and s[1] == -1.0 and s[2] == 1.0 and s[3] == -1.0


def test_unsupported_numbers_penalised():
    made_up = [call("SELECT count(*)"), result("[[12]]"), answer("There are *47* interactions.")]
    real = [call("SELECT count(*)"), result("[[12]]"), answer("There are *12* interactions.")]
    bad, good = groundedness_reward(None, [made_up, real], answer=["12", "12"])
    assert bad < good


def test_correctness_and_validity():
    assert correctness_reward(None, [LEAN], answer=["yes"]) == [1.0]
    bad_call = [{"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "drop_table", "arguments": "{not json"}}]}]
    v = make_validity_reward(["query_biogrid"])(None, [LEAN, bad_call])
    assert v == [1.0, 0.0]


def test_build_reward_funcs():
    cfg = {"weights": {"correctness": 1.0, "efficiency": 0.5, "groundedness": 0.0}}
    funcs, w = build_reward_funcs(cfg, ["query_biogrid"])
    assert [f.__name__ for f in funcs] == ["correctness_reward", "efficiency_reward"] and w == [1.0, 0.5]


def test_rollout_and_metrics_with_scripted_model():
    replies = iter(['[{"tool_name": "query_biogrid", "parameters": {"sql_command": "SELECT 1"}}]',
                    "*Yes*"])

    def query_biogrid(sql_command: str):
        """Fake DB."""
        return [[1]]

    ep = run_episode(lambda msgs: next(replies), [query_biogrid], Q, max_steps=4)
    assert ep["final_answer"] == "*Yes*" and ep["steps"] == 2
    m = episode_metrics(ep["messages"], "Yes")
    assert m["correct"] and m["tool_calls"] == 1 and not m["hallucinated"]
    agg = aggregate([m, episode_metrics(LOOPY, "Yes")])
    assert agg["task_success"] == 1.0 and agg["redundant_call_rate"] == 0.5
