"""SEL-47: the runtime teacher fix, its check, the trainer hand-off and the audit record.

No API, no GPU, no τ²-bench: the teacher model is a fake that returns set turns, and the
classifier is the real one.
"""

from __future__ import annotations

import json
import sys

import pytest

from reflexive.agent import Failure, PassRateGate, ReflexiveAgent
from reflexive.hallucination import classify_record
from reflexive.heal import (
    AUDIT_FORMAT,
    SFT_FORMAT,
    CommandTrainer,
    PrefixTeacher,
    audit_record,
    check_fix,
    fix_point,
    write_audit,
)
from reflexive.splits import task_ids

TRAIN = [("telecom", t) for t in task_ids("telecom", "train")[:3]]
DEV = [("telecom", t) for t in task_ids("telecom", "dev")[:4]]

SYS = {"role": "system", "content": "policy"}
HI = {"role": "user", "content": "my phone 555-123-4567 has no data"}


def call(name, args=None, cid="c1"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args or {})}}
        ],
    }


WRONG_TOOL = call("check_apn_settings")  # a customer-only phone tool: a TOOL failure
GOOD = call("get_customer_by_phone", {"phone_number": "555-123-4567"})
INVENTED = call("get_customer_by_phone", {"phone_number": "555-999-0000"})  # never seen: ARG


def failed_record(domain, task_id, adapter=None):
    view = [SYS, HI, WRONG_TOOL]
    return {
        "domain": domain,
        "task_id": task_id,
        "termination": "user_stop",
        "outcome": {"passed": False},
        "messages": view,
        "agent_view": view,
        "adapter": adapter,
    }


def failure(task=TRAIN[0]):
    rec = failed_record(*task)
    return Failure(rec, classify_record(rec))


class FakeModel:
    """A teacher model that answers with the next turn in ``turns``, whatever it's shown."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []
        self.temperatures = []

    def __call__(self, temperature):
        self.temperatures.append(temperature)
        return self

    def turn(self, messages, tools):
        self.seen.append(messages)
        nxt = self.turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


# --- Where to fix and how a fix is checked ----------------------------------------------------


def test_the_fix_point_is_the_agent_view_up_to_the_first_finding():
    prefix, failed = fix_point(failure())
    assert prefix == [SYS, HI] and failed == WRONG_TOOL


def test_a_fix_is_accepted_only_if_the_classifier_finds_nothing_in_it():
    assert check_fix("telecom", [SYS, HI], GOOD, WRONG_TOOL) == []
    assert any("TOOL" in p for p in check_fix("telecom", [SYS, HI], WRONG_TOOL))
    assert any(p.startswith("ARG") for p in check_fix("telecom", [SYS, HI], INVENTED))
    assert (
        check_fix("telecom", [SYS, HI], WRONG_TOOL, WRONG_TOOL)[0]
        == "the fixed turn is the failed turn again"
    )
    assert check_fix("telecom", [SYS, HI], {"role": "assistant", "content": " "}) == [
        "the fixed turn is empty"
    ]


def say(text):
    return {"role": "assistant", "content": text}


ASK = say("I can help with that. First, could you confirm the phone number on your account?")


def test_a_fix_that_gives_up_is_refused():
    assert check_fix("telecom", [SYS, HI], ASK, WRONG_TOOL) == []  # asking for what's missing is a fix
    assert "the fixed turn ends the conversation" in check_fix(
        "telecom", [SYS, HI], say("I'm sorry, I can't help with that. ###STOP###")
    )
    assert "the fixed turn hands off to a human" in check_fix(
        "telecom", [SYS, HI], call("transfer_to_human_agents", {"summary": "no data"})
    )
    for text in ("OK.", "Sure.", "###STOP###"):
        assert any(
            "too short" in p or "ends the conversation" in p
            for p in check_fix("telecom", [SYS, HI], say(text))
        )
    assert check_fix("telecom", [SYS, HI], say("Could you please provide your email address?")) == []


# --- The teacher -------------------------------------------------------------------------------


def test_the_teacher_sees_only_the_prefix_and_its_fix_trains_on_that_turn_only():
    model = FakeModel([GOOD])
    teacher = PrefixTeacher(make_agent=model)
    ex = teacher.fix(failure())
    assert ex is not None

    assert model.seen == [[SYS, HI]]  # nothing from the failing turn on
    assert ex["format"] == SFT_FORMAT and ex["labels"] == ["TOOL"]
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert ex["train_on"] == [2]  # only the fixed turn, never the failed conversation's turns
    assert "get_customer_by_phone" in ex["messages"][2]["content"]
    assert "Action:" in ex["messages"][2]["content"]  # Aya's own format, as golden records
    assert ex["source"]["fixed_turn"] == 2 and ex["source"]["attempt"] == 1


def test_the_teacher_retries_until_a_fix_passes_the_check():
    model = FakeModel([WRONG_TOOL, RuntimeError("rate limited"), GOOD])
    teacher = PrefixTeacher(make_agent=model)
    ex = teacher.fix(failure())
    assert ex is not None

    assert ex["source"]["attempt"] == 3 and model.temperatures == [0.0, 0.7, 0.7]
    log = teacher.tried[TRAIN[0]]
    assert [a["accepted"] for a in log] == [False, False, True]
    assert "failed turn again" in log[0]["problems"][0] and "rate limited" in log[1]["error"]


def test_the_teacher_gives_up_after_its_attempts():
    teacher = PrefixTeacher(make_agent=FakeModel([INVENTED] * 3))
    assert teacher.fix(failure()) is None and len(teacher.tried[TRAIN[0]]) == 3


def test_the_teacher_never_fixes_a_dev_task():
    with pytest.raises(ValueError, match="train"):
        PrefixTeacher(make_agent=FakeModel([GOOD])).fix(failure(DEV[0]))


# --- The trainer -------------------------------------------------------------------------------


def test_the_trainer_writes_the_fixes_and_runs_its_command(tmp_path):
    ex = PrefixTeacher(make_agent=FakeModel([GOOD])).fix(failure())
    assert ex is not None
    cmd = [
        sys.executable,
        "-c",
        "import sys, pathlib; pathlib.Path(sys.argv[2], 'adapter').write_text(sys.argv[1])",
        "{data}",
        "{out}",
    ]
    trainer = CommandTrainer(tmp_path, command=cmd)

    name = trainer.train([ex], None)

    out = tmp_path / name
    assert [json.loads(line) for line in (out / "fixes.jsonl").read_text().splitlines()] == [ex]
    assert (out / "adapter").read_text() == str(out / "fixes.jsonl")
    assert trainer.calls[0]["returncode"] == 0 and trainer.calls[0]["examples"] == 1


def test_a_failed_training_run_stops_the_cycle(tmp_path):
    ex = PrefixTeacher(make_agent=FakeModel([GOOD])).fix(failure())
    assert ex is not None
    trainer = CommandTrainer(tmp_path, command=[sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(RuntimeError, match="exit 3"):
        trainer.train([ex], None)


def test_the_trainer_refuses_anything_but_sft_examples(tmp_path):
    with pytest.raises(ValueError, match=SFT_FORMAT):
        CommandTrainer(tmp_path).train([{"format": "stub.fix", "id": "x"}], None)


# --- A whole cycle with the real parts, and its audit record ----------------------------------


def test_a_cycle_with_the_real_teacher_and_trainer_writes_its_audit_record(tmp_path):
    def rollout(adapter, domain, task_id):
        if (domain, task_id) in TRAIN:
            return failed_record(domain, task_id, adapter)
        ok = adapter is not None and (domain, task_id) in DEV[:2]
        view = [SYS, HI, {"role": "assistant", "content": "Done."}]
        return {"domain": domain, "task_id": task_id, "termination": "user_stop", "outcome": {"passed": ok},
                "messages": view, "agent_view": view, "adapter": adapter}  # fmt: skip

    # task 1 fixed at once; task 2 never (an invented number every time); task 3 on the retry
    teacher = PrefixTeacher(make_agent=FakeModel([GOOD, INVENTED, INVENTED, INVENTED, WRONG_TOOL, GOOD]))
    trainer = CommandTrainer(tmp_path / "runs")  # dry run: writes the data, trains nothing
    agent = ReflexiveAgent(rollout, teacher=teacher, trainer=trainer, gate=PassRateGate())

    report = agent.run_cycle(TRAIN, DEV)
    rec = write_audit(report, tmp_path / "audit.jsonl", cycle=1, teacher=teacher, trainer=trainer)

    assert report.swapped and agent.adapter == report.candidate
    assert json.loads((tmp_path / "audit.jsonl").read_text()) == json.loads(json.dumps(rec))
    assert rec["format"] == AUDIT_FORMAT and rec["cycle"] == 1
    assert rec["adapter"] == {
        "before": None,
        "candidate": report.candidate,
        "after": report.candidate,
        "swapped": True,
    }
    assert [f["fixed"] for f in rec["failures"]] == [True, False, True]
    assert [len(f["attempts"]) for f in rec["failures"]] == [1, 3, 2]
    assert (
        rec["failures"][0]["failure"] == "TOOL"
        and rec["failures"][0]["first"]["tool"] == "check_apn_settings"
    )
    assert rec["fixes"]["accepted"] == 2 and rec["fixes"]["of"] == 3
    assert rec["training"]["examples"] == 2 and rec["training"]["base"] is None
    assert rec["gate"]["passed"] and len(rec["gate"]["tasks"]) == len(DEV)


def test_a_skipped_cycle_still_has_an_audit_record():
    def rollout(adapter, domain, task_id):
        view = [SYS, HI, {"role": "assistant", "content": "Done."}]
        return {"domain": domain, "task_id": task_id, "termination": "user_stop", "outcome": {"passed": True},
                "messages": view, "agent_view": view}  # fmt: skip

    agent = ReflexiveAgent(
        rollout,
        teacher=PrefixTeacher(make_agent=FakeModel([])),
        trainer=CommandTrainer(),
        gate=PassRateGate(),
    )
    rec = audit_record(agent.run_cycle(TRAIN, DEV), cycle=2)
    assert rec["skipped"] == "no failures to heal" and rec["gate"] is None and rec["training"] is None
    assert rec["adapter"]["swapped"] is False and rec["heal"]["passed"] == 3


def test_near_duplicate_fixes_are_capped_per_lesson(tmp_path):
    from reflexive.heal import dedupe, lesson

    asks = {"role": "assistant", "content": "Could you give me your phone number?"}
    fixes = []
    for task in TRAIN:
        ex = PrefixTeacher(make_agent=FakeModel([asks])).fix(failure(task))
        assert ex is not None
        fixes.append(ex)
    good = PrefixTeacher(make_agent=FakeModel([GOOD])).fix(failure())
    assert good is not None
    fixes.append(good)

    assert lesson(fixes[0]) == "telecom:TOOL/customer_tool/check_apn_settings -> message"
    assert lesson(good).endswith("-> get_customer_by_phone")
    kept, dropped = dedupe(fixes, 2)
    assert kept == [fixes[0], fixes[1], good] and dropped == [fixes[2]]
    assert dedupe(fixes, None) == (fixes, [])

    trainer = CommandTrainer(tmp_path)
    trainer.train(fixes, None)
    call = trainer.calls[0]
    assert call["examples"] == 3 and call["dropped_duplicates"] == [fixes[2]["id"]]
    assert sorted(call["lessons"].values()) == [1, 2]


def test_run_metrics_sum_up_a_cycle_from_its_audit_and_conversations(tmp_path):
    from reflexive.heal import METRICS_FORMAT, run_metrics

    def rollout(adapter, domain, task_id):
        rec = failed_record(domain, task_id, adapter)
        rec["agent_usage"], rec["user_cost"] = {"est_cost": 0.10}, 0.05
        return rec

    # task 1 fixed at once, task 2 never, task 3 at once; the fake has no usage: a cost is set below
    teacher = PrefixTeacher(make_agent=FakeModel([GOOD, INVENTED, INVENTED, INVENTED, GOOD]))
    agent = ReflexiveAgent(rollout, teacher=teacher, trainer=CommandTrainer(tmp_path), gate=PassRateGate())
    report = agent.run_cycle(TRAIN, DEV, stop_after_fixes=True)
    audit = audit_record(report, cycle=1, teacher=teacher)
    audit["failures"][0]["attempts"][0]["usage"] = {"billed_cost": 0.001}
    episodes = [{**r, "heal_role": "heal", "cycle": 1} for r in report.rollouts]

    m = run_metrics([audit], episodes, {"sample": 3})

    assert m["format"] == METRICS_FORMAT and m["run"] == {"sample": 3} and m["conversations"] == 3
    (c,) = m["cycles"]
    assert (c["passed"], c["conversations"], c["failures"], c["fixable"], c["fixed"]) == (0, 3, 3, 3, 2)
    assert c["first_try"] == 2 and c["fixed_at_turn"] == {"2": 2}
    assert c["failure_types"] == {"TOOL": 3, "ARG": 0, "LOOP": 0, "NONE": 0}
    assert [t["attempts"] for t in c["tasks"]] == [1, 3, 1] and c["tasks"][0]["termination"] == "user_stop"
    assert m["cost_usd"] == {
        "student": 0.3,
        "customer": 0.15,
        "teacher": 0.001,
        "total": 0.451,
        "note": "judge calls not included",
    }
