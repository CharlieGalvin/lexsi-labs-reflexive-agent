"""SEL-75: heal GRPO on the teacher's fixes at the student's Tool / Arg / Loop failures.

Synthetic conversations in the agent's view (OpenAI format), so the tests need no golden data:
one per failure type, built from the real tool schemas. The teacher is a stub; the rewards,
the data build and the reward check run for real.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest
from _scripts import load_script

from reflexive import heal_data, heal_rewards
from reflexive.teacher import to_prompted

mf = load_script("heal/make_fixes")
build_script = load_script("data/build_grpo_heal_data")
check_script = load_script("data/check_heal_rewards")

Msg = dict[str, Any]


def call(i: str, name: str, **args: Any) -> Msg:
    return {"id": i, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def agent(*calls: Msg, text: str = "") -> Msg:
    m: Msg = {"role": "assistant", "content": text}
    if calls:
        m["tool_calls"] = list(calls)
    return m


def result(i: str, content: str) -> Msg:
    return {"role": "tool", "tool_call_id": i, "content": content}


SYSTEM = {"role": "system", "content": "You are a customer service agent. Follow the policy."}
GREET = agent(text="Hi! How can I help you today?")


def retail_failure(task_id: str = "0") -> dict[str, Any]:
    """Arg at turn 3 (a placeholder ID), then a Loop at turn 5 (the same failed call again)."""
    msgs = [
        SYSTEM,
        GREET,
        {"role": "user", "content": "I want to return my order. My email is yusuf.rossi@example.com."},
        agent(call("c1", "get_order_details", order_id="<order_id>")),
        result("c1", "Error: Order not found"),
        agent(call("c2", "get_order_details", order_id="<order_id>")),
        result("c2", "Error: Order not found"),
    ]
    return {"domain": "retail", "task_id": task_id, "group": "healable", "messages": msgs}


def telecom_failure(task_id: str = "0") -> dict[str, Any]:
    """Tool at turn 3: the agent calls the customer's own phone tool."""
    msgs = [
        SYSTEM,
        GREET,
        {"role": "user", "content": "My mobile data isn't working. My number is 555-123-2002."},
        agent(call("c1", "check_apn_settings")),
        result("c1", "Error: Tool not found"),
    ]
    return {"domain": "telecom", "task_id": task_id, "group": "healable", "messages": msgs}


def train_task(domain: str, k: int = 0) -> str:
    from reflexive import splits

    return sorted(splits.task_ids(domain, "train"), key=str)[k]


# --- Points ---------------------------------------------------------------------------------------


def test_each_failure_type_gives_its_own_point_at_its_first_finding():
    pts = heal_data.points_of(retail_failure())
    assert [(types, f.turn, f.kind) for types, f in pts] == [
        (["ARG"], 3, "placeholder"),
        (["LOOP"], 5, "repeated_failed_call"),
    ]
    ((types, f),) = heal_data.points_of(telecom_failure())
    assert types == ["TOOL"] and f.kind == "customer_tool" and f.turn == 3


def test_a_point_cuts_the_prefix_at_its_own_finding():
    types, finding = heal_data.points_of(retail_failure())[1]
    diag = heal_data.diagnosis_for(types, finding)
    assert diag.failure.value == "LOOP" and diag.first is not None and diag.first.turn == 5


def test_only_healable_train_tasks_are_fixed(tmp_path: Path):
    t = train_task("retail")
    hard = [
        {**retail_failure(t), "group": "healable"},
        {**retail_failure(t), "group": "solved_by_aya"},
    ]
    recs = mf.failed_records(hard, [], mf.allowed_tasks(hard))
    assert len(recs) == 1 and recs[0]["point_source"] == "armA"
    ep = tmp_path / "S2-retail-trainpool.jsonl"
    rows = [
        {**retail_failure(t), "agent_view": retail_failure(t)["messages"], "outcome": {"passed": False}},
        {**retail_failure(t), "outcome": {"passed": True}},  # passed: nothing to heal
        {**retail_failure(train_task("retail", 1)), "outcome": {"passed": False}},  # not a healable task
    ]
    ep.write_text("".join(json.dumps(r) + "\n" for r in rows))
    recs = mf.failed_records(hard, [ep], mf.allowed_tasks(hard))
    assert [r["point_source"] for r in recs] == ["armA", "S2-retail-trainpool"]


# --- Fixes, with a stub teacher -----------------------------------------------------------------


class Stub:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}

    def __init__(self, turn: Msg) -> None:
        self._turn = turn

    def turn(self, messages: list[Msg], tools: list[Msg]) -> Msg:
        return self._turn


ASK = agent(text="Could you tell me the order ID? It starts with # and is on your confirmation email.")


def fixes_for(record: dict[str, Any], turn: Msg = ASK) -> list[dict[str, Any]]:
    out = []
    for pid, rec, types, f in mf.points([{**record, "point_source": "armA"}]):
        log = mf.fix_one(pid, rec, types, f, {"make_agent": lambda t: Stub(turn)})
        assert log["accepted"], log["attempts"]
        out.append(log["example"])
    return out


def test_a_fix_is_labelled_with_its_point_and_keeps_the_failed_turn():
    t = train_task("retail")
    arg, loop = fixes_for(retail_failure(t))
    assert arg["labels"] == ["ARG"] and loop["labels"] == ["LOOP"]
    assert arg["id"] == f"heal/armA/retail/{t}/s0/t3" and arg["source"]["point_source"] == "armA"
    assert arg["train_on"] == [len(arg["messages"]) - 1]
    assert heal_data.failed_of(loop)["calls"] == [
        heal_data.call_key("get_order_details", {"order_id": "<order_id>"})
    ]


def test_a_fix_that_repeats_the_failure_is_refused():
    t = train_task("retail")
    ((pid, rec, types, f), _) = mf.points([{**retail_failure(t), "point_source": "armA"}])
    bad = agent(call("x", "get_order_details", order_id="<order_id>"))
    log = mf.fix_one(pid, rec, types, f, {"make_agent": lambda _: Stub(bad)})
    assert not log["accepted"] and "example" not in log
    assert any(
        "failed turn again" in p or "ARG/placeholder" in p for a in log["attempts"] for p in a["problems"]
    )


def test_the_script_fixes_resumes_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    t = train_task("retail")
    hard = tmp_path / "hard.jsonl"
    hard.write_text(json.dumps(retail_failure(t)) + "\n")
    from reflexive.heal import PrefixTeacher

    monkeypatch.setattr(mf, "PrefixTeacher", functools.partial(PrefixTeacher, make_agent=lambda _: Stub(ASK)))
    out = tmp_path / "heal"
    assert mf.main(["--hard", str(hard), "--out", str(out), "--workers", "2"]) == 0
    fixes = [json.loads(line) for line in (out / "fixes.jsonl").read_text().splitlines()]
    assert sorted(f["labels"][0] for f in fixes) == ["ARG", "LOOP"]
    rep = json.loads((out / "fixes.report.json").read_text())
    assert rep["accepted"] == 2 and rep["by_type"]["LOOP"] == {"points": 1, "accepted": 1}
    assert mf.main(["--hard", str(hard), "--out", str(out)]) == 0  # resumed: nothing new
    assert len((out / "attempts.jsonl").read_text().splitlines()) == 2


# --- GRPO rows ----------------------------------------------------------------------------------


def golden(task_id: str) -> dict[str, Any]:
    """A teacher conversation in ``reagent.sft/v1``: a reply and a tool call as targets."""
    from reflexive.hallucination import domain_tools

    tools, _ = domain_tools("retail")
    view = [
        SYSTEM,
        GREET,
        {"role": "user", "content": "I'd like to check order #W2378156."},
        agent(call("g1", "get_order_details", order_id="#W2378156")),
        result("g1", '{"order_id": "#W2378156", "status": "delivered"}'),
        agent(text="Your order #W2378156 was delivered. Would you like to return any items?"),
    ]
    msgs = [{"role": m["role"], "content": m["content"]} for m in to_prompted(view, tools, "cohere")]
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][1:]
    return {
        "id": f"glm53f/retail/{task_id}/1",
        "format": "reagent.sft/v1",
        "domain": "retail",
        "task_id": task_id,
        "group": "healable",
        "labels": ["ARG"],
        "messages": msgs,
        "train_on": asst,
        "source": {},
    }


def test_rows_carry_the_heal_columns_and_the_prompt_history():
    t = train_task("retail")
    arg, loop = fixes_for(retail_failure(t))
    (row,) = heal_data.turn_rows(loop)
    assert row["failure_type"] == "LOOP" and row["kind"] == "text"
    assert row["history"]["calls"] == [
        {"key": heal_data.call_key("get_order_details", {"order_id": "<order_id>"}), "failed": True}
    ]
    anchors = heal_data.turn_rows(golden(t), heal_data.NONE)
    assert [r["kind"] for r in anchors] == ["call", "text"] and {r["failure_type"] for r in anchors} == {
        "NONE"
    }
    assert anchors[1]["history"]["calls"][0]["failed"] is False


def test_build_holds_out_whole_tasks_and_mixes_anchors_of_both_kinds():
    tasks = [train_task("retail", k) for k in range(10)]
    fixes = [f for t in tasks for f in fixes_for(retail_failure(t))]
    train, held, report = heal_data.build(fixes, [golden(t) for t in tasks], anchors_per_fix=0.5, seed=1)
    assert report["heldout"]["tasks"] == 1 and len(report["heldout_tasks"]) == 1
    held_tasks = {r["task_id"] for r in held}
    assert held_tasks.isdisjoint({r["task_id"] for r in train})
    fix_rows = [r for r in train if r["failure_type"] != "NONE"]
    assert len([r for r in train if r["failure_type"] == "NONE"]) == round(0.5 * len(fix_rows))


def test_build_refuses_dev_tasks_unlabelled_fixes_and_no_fixes():
    t = train_task("retail")
    fixes = fixes_for(retail_failure(t))
    from reflexive import splits

    dev = sorted(splits.task_ids("retail", "dev"), key=str)[0]
    with pytest.raises(heal_data.HealDataError) as e:
        heal_data.build(fixes, [golden(dev)])
    assert any("dev" in line for line in e.value.lines), e.value.lines
    with pytest.raises(heal_data.HealDataError) as e:
        heal_data.build([{**fixes[0], "labels": []}], [])
    assert any("no Tool / Arg / Loop label" in line for line in e.value.lines)
    with pytest.raises(heal_data.HealDataError, match="no fixes"):
        heal_data.build([], [golden(t)])


# --- Rewards ------------------------------------------------------------------------------------


def action(*calls: tuple[str, dict]) -> str:
    body = [{"tool_name": n, "parameters": a} for n, a in calls]
    return f"Action: ```json\n{json.dumps(body, indent=4)}\n```"


def row_for(fix: dict[str, Any]) -> dict[str, Any]:
    (row,) = heal_data.turn_rows(fix)
    return row


# Fixes that are tool calls (grounded in the conversation), for the failure rewards to score against
FIND_YUSUF = agent(call("f1", "find_user_id_by_email", email="yusuf.rossi@example.com"))
PHONE = agent(call("f1", "get_customer_by_phone", phone_number="555-123-2002"))


def test_each_failure_type_has_its_own_reward():
    t = train_task("retail")
    arg_fix, loop_fix = fixes_for(retail_failure(t), FIND_YUSUF)
    (tool_fix,) = fixes_for(telecom_failure(train_task("telecom")), PHONE)
    tool = heal_rewards.score_row(action(("check_apn_settings", {})), row_for(tool_fix))
    assert tool["tool"] == 0.0 and tool["customer_tools"] == ["check_apn_settings"]
    arg = heal_rewards.score_row(action(("get_order_details", {"order_id": "<order_id>"})), row_for(arg_fix))
    assert arg["arg"] == 0.0 and arg["arg_kinds"] == ["placeholder"] and arg["tool"] == 1.0
    loop = heal_rewards.score_row(
        action(("get_order_details", {"order_id": "<order_id>"})), row_for(loop_fix)
    )
    assert loop["loop"] == 0.0 and "repeated_failed_call" in loop["loop_kinds"]
    right = heal_rewards.score_row(
        action(("find_user_id_by_email", {"email": "yusuf.rossi@example.com"})), row_for(arg_fix)
    )
    assert heal_rewards.total(right, heal_rewards.DEFAULT_WEIGHTS) == 4.0


def test_a_tool_docstring_example_counts_as_invented():
    t = train_task("retail")
    arg_fix, _ = fixes_for(retail_failure(t), FIND_YUSUF)
    s = heal_rewards.score_row(
        action(("find_user_id_by_email", {"email": "something@example.com"})), row_for(arg_fix)
    )
    assert s["arg"] == 0.0 and s["arg_kinds"] == ["ungrounded"]


def test_the_failure_rewards_count_only_for_the_teachers_kind_of_turn():
    """Chiara's review: a reply can't fail tool / arg / loop, so without this it scored 3.0 of 4.0
    where the teacher called, more than a call with one invented argument."""
    t = train_task("retail")
    call_row, text_row = heal_data.turn_rows(golden(t), heal_data.NONE)
    w = heal_rewards.DEFAULT_WEIGHTS
    generic = heal_rewards.score_row("Could you please provide more details?", call_row)
    assert (generic["tool"], generic["arg"], generic["loop"], generic["fix"]) == (0.0, 0.0, 0.0, 0.0)
    invented = heal_rewards.score_row(action(("get_order_details", {"order_id": "#W9999999"})), call_row)
    assert heal_rewards.total(invented, w) > heal_rewards.total(
        generic, w
    )  # the right kind of turn beats a dodge
    call_on_text = heal_rewards.score_row(action(("get_order_details", {"order_id": "#W2378156"})), text_row)
    assert call_on_text["tool"] == call_on_text["fix"] == 0.0


def test_what_a_reply_says_counts_where_the_teacher_replied():
    t = train_task("retail")
    _, text_row = heal_data.turn_rows(golden(t), heal_data.NONE)
    teacher = heal_rewards.score_row(text_row["target"], text_row)
    close = heal_rewards.score_row(
        "Order #W2378156 was delivered. Do you want to return any items?", text_row
    )
    generic = heal_rewards.score_row("Could you please provide more details?", text_row)
    assert teacher["fix"] == 1.0 > close["fix"] > generic["fix"] == heal_rewards.REPLY_KIND
    filler = "Let me look into that. One moment, please."
    assert heal_rewards.word_overlap(filler, "Could you provide details?") == 1.0  # both say nothing
    assert heal_rewards.word_overlap("Your order #W2378156 shipped.", "Could you provide details?") == 0.5


def test_the_third_identical_call_and_message_are_loops_but_the_teacher_may_repeat():
    hist = {
        "calls": [{"key": heal_data.call_key("calculate", {"expression": "2+2"}), "failed": False}] * 2,
        "texts": [heal_data.norm_text("Please hold on.")] * 2,
    }
    kw: dict[str, Any] = {
        "target": "Sure.",
        "target_calls": [],
        "domain": "retail",
        "prompt": [],
        "history": hist,
    }
    assert heal_rewards.score(action(("calculate", {"expression": "2+2"})), **kw)["loop_kinds"] == [
        "repeated_call"
    ]
    assert heal_rewards.score("Please hold on.", **kw)["loop_kinds"] == ["repeated_message"]
    kw["target_calls"] = [{"name": "calculate", "arguments": {"expression": "2+2"}}]  # the teacher repeats it
    assert heal_rewards.score(action(("calculate", {"expression": "2+2"})), **kw)["loop"] == 1.0


def test_the_reward_functions_score_a_batch_once_and_log_every_sample(tmp_path: Path):
    t = train_task("retail")
    arg_fix, _ = fixes_for(retail_failure(t))
    row = row_for(arg_fix)
    cols = {
        k: [json.dumps(row[k]) if k in ("target_calls", "failed", "history") else row[k]] * 2
        for k in ("target", "target_calls", "domain", "id", "failure_type", "failed", "history")
    }
    rf = heal_rewards.RewardFns(tmp_path / "rewards.jsonl")
    assert [getattr(f, "__name__", None) for f in rf.fns] == list(
        heal_rewards.COMPONENTS
    )  # TRL names charts by it
    completions = [ASK["content"], action(("get_order_details", {"order_id": "<order_id>"}))]
    seen = {}
    outs = [
        f(completions=completions, prompts=[row["prompt"]] * 2, log_metric=seen.__setitem__, **cols)
        for f in rf.fns
    ]
    assert outs[0] == [1.0, 0.0] and outs[1] == [
        1.0,
        0.0,
    ]  # tool, then arg: the call isn't the teacher's kind
    assert rf.calls == 1  # scored once for all five
    logged = [json.loads(x) for x in (tmp_path / "rewards.jsonl").read_text().splitlines()]
    assert len(logged) == 2 and logged[0]["failure_type"] == "ARG" and logged[0]["reward"] == 4.0
    assert seen["heal/reply_share"] == 0.5 and seen["heal/reply_excess"] == -0.5  # the teacher replied twice


def test_the_reward_check_script_on_built_data(tmp_path: Path, capsys: pytest.CaptureFixture):
    tasks = [train_task("retail", k) for k in range(4)]
    fixes = [f for t in tasks for f in fixes_for(retail_failure(t))]
    train, _, _ = heal_data.build(fixes, [golden(t) for t in tasks], heldout_fraction=0)
    data = tmp_path / "grpo-heal.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in train))
    assert check_script.main(["--data", str(data)]) == 0
    out = capsys.readouterr().out
    assert "canned replies" in out
    assert "fails the arg reward at 4/4" in out and "fails the loop reward at 4/4" in out


def test_the_data_script_writes_both_files(tmp_path: Path):
    tasks = [train_task("retail", k) for k in range(4)]
    fixes = tmp_path / "fixes.jsonl"
    fixes.write_text("".join(json.dumps(f) + "\n" for t in tasks for f in fixes_for(retail_failure(t))))
    gold = tmp_path / "golden.jsonl"
    gold.write_text("".join(json.dumps(golden(t)) + "\n" for t in tasks))
    assert build_script.main(["--fixes", str(fixes), "--golden", str(gold), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "grpo-heal.heldout.jsonl").exists()
    rep = json.loads((tmp_path / "grpo-heal.report.json").read_text())
    assert rep["train"]["failure_type"]["ARG"] >= 1 and rep["format"] == heal_data.FORMAT


# --- The trainer and the held-out check with the heal rewards ------------------------------------

ts = load_script("grpo/train_single_turn")
hc = load_script("grpo/heldout_check")
HEAL_CONFIG = ts.ROOT / "configs" / "train" / "grpo_heal_aya32b.yaml"


def heal_rows(n_tasks: int = 2) -> list[dict[str, Any]]:
    tasks = [train_task("retail", k) for k in range(n_tasks)]
    fixes = [f for t in tasks for f in fixes_for(retail_failure(t))]
    train, _, _ = heal_data.build(fixes, [golden(t) for t in tasks], heldout_fraction=0)
    return train


def heal_cfg(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path = tmp_path / "grpo-heal.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {**ts.load(HEAL_CONFIG), "data": str(path)}


def test_the_heal_config_differs_from_g_only_in_data_and_rewards():
    heal, g = ts.load(HEAL_CONFIG), ts.load(ts.CONFIG)
    assert ts.reward_set(heal) is heal_rewards and ts.reward_set(g).COMPONENTS != heal_rewards.COMPONENTS
    assert heal["trainer"] == g["trainer"] and heal["adapter"] == g["adapter"] and heal["model"] == g["model"]
    w = heal["reward_weights"]
    assert w["tool"] == w["arg"] > w["loop"]  # optimise for Tool and Arg


def test_unknown_reward_sets_and_components_are_refused(tmp_path: Path):
    path = tmp_path / "c.yaml"
    head = "name: x\nmodel: m\nadapter: a\ndata: d\ntrainer: {}\n"
    path.write_text(head + "rewards: heal\nreward_weights: {tool_match: 1}\n")  # the first run's component
    with pytest.raises(SystemExit, match="unknown reward component"):
        ts.load(path)
    path.write_text(head + "rewards: magic\n")
    with pytest.raises(SystemExit, match="unknown rewards"):
        ts.load(path)


def test_the_heal_dataset_columns_are_json_where_arrow_needs_it(tmp_path: Path):
    rows = ts.prompts(heal_cfg(tmp_path, heal_rows()))
    assert set(rows[0]) == set(ts.HEAL_COLUMNS)
    assert all(isinstance(r[c], str) for r in rows for c in ts.JSON_COLUMNS)
    first_run_rows = [{k: v for k, v in r.items() if k not in ("failed", "history")} for r in heal_rows()]
    with pytest.raises(SystemExit, match="grpo-heal-data"):
        ts.prompts(heal_cfg(tmp_path, first_run_rows))


def test_the_heldout_check_scores_with_the_heal_rewards_and_reports_kind_match():
    rows = heal_rows()
    texts = [[r["target"], "Thanks, one moment while I look into that for you right now."] for r in rows]
    scored = hc.score_samples(rows, texts, heal_rewards.DEFAULT_WEIGHTS, heal_rewards)
    summary = hc.summarise({"sft-v2": scored}, 4.0, heal_rewards.COMPONENTS)
    a = summary["adapters"]["sft-v2"]
    calls = sum(r["kind"] == "call" for r in rows)
    assert a["kind_match"] == pytest.approx(1 - calls / (2 * len(rows)), abs=1e-3)  # replies miss the calls
    assert set(a["by_failure_type"]) == {"ARG", "LOOP", "NONE"}
    md = hc.to_markdown(summary)
    assert "kind_match" in md and "| sft-v2 | ARG |" in md and "tool_match" not in md.split("\n")[0]


def test_one_real_grpo_step_with_the_heal_rewards(tmp_path: Path):
    pytest.importorskip("torch")
    pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    import _tiny
    from datasets import Dataset

    tok = _tiny.tokenizer()
    model = _tiny.adapter_model(tmp_path, tok, ts.prepare_for_training)
    cfg = heal_cfg(tmp_path, heal_rows()[:2])
    rows = ts.prompts(cfg)
    weights = {**heal_rewards.DEFAULT_WEIGHTS, **cfg["reward_weights"]}
    args = ts.grpo_args(
        cfg,
        tmp_path / "run",
        {
            "max_steps": 1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "steps_per_generation": 2,
            "num_generations": 4,
            "max_completion_length": 8,
            "optim": "adamw_torch",
            "report_to": [],
            "bf16": False,
            "save_strategy": "no",
            "use_cpu": True,
            "reward_weights": [weights[c] for c in heal_rewards.COMPONENTS],
        },
    )
    rf = heal_rewards.RewardFns(tmp_path / "rewards.jsonl", weights)
    trainer = ts.releasing_trainer(trl.GRPOTrainer)(
        model=model,
        reward_funcs=rf.fns,
        args=trl.GRPOConfig(**args),
        train_dataset=Dataset.from_list(rows),
        processing_class=tok,
    )
    trainer.train()
    logged = [json.loads(x) for x in (tmp_path / "rewards.jsonl").read_text().splitlines()]
    assert len(logged) == 8 and rf.calls == 2  # two rounds of one group, each scored once for all five
    assert {r["failure_type"] for r in logged} <= {"ARG", "LOOP", "NONE"}
    metrics = trainer.state.log_history[0]
    assert all(f"rewards/{c}/mean" in metrics for c in heal_rewards.COMPONENTS)
    assert "heal/reply_excess" in metrics  # what the drift guard watches


def test_the_reward_check_fails_when_a_canned_reply_scores_too_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    t = train_task("retail")
    train, _, _ = heal_data.build(
        fixes_for(retail_failure(t)), [golden(t)], heldout_fraction=0, anchors_per_fix=1
    )
    data = tmp_path / "grpo-heal.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in train))
    monkeypatch.setattr(check_script, "CANNED_WHERE_CALLED", 0.0)  # a canned reply gets 0.5 there (guard)
    assert check_script.main(["--data", str(data)]) == 1
    assert "a canned reply scores too well" in capsys.readouterr().out


def test_the_drift_guard_stops_when_replies_drift_from_the_teacher():
    guard = ts.ReplyDrift(max_drift=0.10, window=3, after_step=5)
    steps = [0.05, -0.05, 0.0, 0.05, 0.0, 0.1, 0.05]  # baseline ~0, small wobble: carry on
    assert all(guard.update(i + 1, {"heal/reply_excess": v}) is None for i, v in enumerate(steps))
    why = None
    for i, v in enumerate([0.3, 0.3, 0.3]):  # replying 30 points more than the teacher
        why = why or guard.update(len(steps) + i + 1, {"heal/reply_excess": v})
    assert why and why.startswith("step 8:")  # the first step that pushes the average over
    assert guard.update(99, {"heal/reply_excess": 0.0}) == why  # stays stopped
    g = ts.ReplyDrift(window=2, after_step=1)
    for i, v in enumerate([-0.2, -0.2, -0.2, -0.45, -0.45]):  # G's way: far fewer replies
        g.update(i + 1, {"heal/reply_excess": v})
    assert g.stopped and g.update(9, {"loss": 1.0}) == g.stopped
