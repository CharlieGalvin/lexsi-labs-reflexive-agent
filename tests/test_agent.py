"""SEL-40: one ReflexiveAgent heal cycle end to end, with a fake rollout and the stub parts.

No GPU, no τ²-bench, no API: the rollout returns built records, and the classifier is the real one.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from reflexive.agent import PassRateGate, ReflexiveAgent, StubTeacher, StubTrainer, arm_rollout
from reflexive.hallucination import FailureType
from reflexive.splits import task_ids

TRAIN = [("telecom", t) for t in task_ids("telecom", "train")[:3]]
DEV = [("telecom", t) for t in task_ids("telecom", "dev")[:4]]
TEST = ("telecom", task_ids("telecom", "test")[0])

SYS = {"role": "system", "content": "policy"}
HI = {"role": "user", "content": "my phone has no data"}
# The agent calls one of the customer's phone tools itself: a TOOL failure
WRONG_TOOL = {
    "role": "assistant",
    "content": "",
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "check_apn_settings", "arguments": "{}"}}
    ],
}


def record(domain, task_id, ok, adapter):
    view = [SYS, HI, {"role": "assistant", "content": "Done."}] if ok else [SYS, HI, WRONG_TOOL]
    return {
        "domain": domain,
        "task_id": task_id,
        "termination": "user_stop",
        "outcome": {"passed": ok},
        "messages": view,
        "agent_view": view,
        "adapter": adapter,
    }


class FakeRollout:
    """The base student fails every task; an adapter passes the dev tasks in ``heals``."""

    def __init__(self, heals=frozenset()):
        self.heals = heals
        self.calls = []

    def __call__(self, adapter, domain, task_id):
        self.calls.append((adapter, domain, task_id))
        return record(domain, task_id, adapter is not None and (domain, task_id) in self.heals, adapter)


def make(rollout, **kw):
    teacher, trainer = StubTeacher(**kw), StubTrainer()
    return ReflexiveAgent(rollout, teacher=teacher, trainer=trainer, gate=PassRateGate()), teacher, trainer


def test_a_cycle_runs_end_to_end_and_swaps_when_the_gate_passes():
    rollout = FakeRollout(heals=set(DEV[:2]))
    agent, teacher, trainer = make(rollout)

    report = agent.run_cycle(TRAIN, DEV)

    assert [f.diagnosis.failure for f in report.failures] == [FailureType.TOOL] * 3
    assert len(teacher.seen) == 3 and len(report.fixes) == 3
    assert trainer.calls == [(report.fixes, None)]  # trained from the base student on the fixes
    assert report.gate.passed and (report.gate.baseline_rate, report.gate.candidate_rate) == (0.0, 0.5)
    assert agent.adapter == report.candidate == "stub-adapter-1" and report.swapped
    # heal tasks ran on the old adapter; the gate ran both adapters on dev
    assert [c[0] for c in rollout.calls] == [None] * 3 + [None] * 4 + ["stub-adapter-1"] * 4
    assert report.summary()["failures"]["TOOL"] == 3 and agent.history == [report]


def test_the_adapter_is_kept_when_the_gate_fails():
    agent, _, _ = make(FakeRollout())  # the candidate is no better on dev
    report = agent.run_cycle(TRAIN, DEV)
    assert report.gate is not None and not report.gate.passed
    assert agent.adapter is None and not report.swapped and report.candidate == "stub-adapter-1"


def test_failures_the_teacher_cant_fix_are_dropped():
    agent, _, trainer = make(FakeRollout(heals=set(DEV)), give_up_on={TRAIN[0]})
    report = agent.run_cycle(TRAIN, DEV)
    assert len(report.failures) == 3 and len(report.fixes) == 2
    assert [fx["task_id"] for fx in trainer.calls[0][0]] == [t for _, t in TRAIN[1:]]


def test_a_cycle_with_nothing_to_train_on_stops_before_retraining():
    agent, _, trainer = make(FakeRollout(), give_up_on=set(TRAIN))
    report = agent.run_cycle(TRAIN, DEV)
    assert report.skipped == "the teacher fixed none of the failures"
    assert trainer.calls == [] and report.gate is None and agent.adapter is None


def test_no_failures_means_no_heal():
    agent, teacher, trainer = make(lambda a, d, t: record(d, t, True, a))
    report = agent.run_cycle(TRAIN, DEV)
    assert report.skipped == "no failures to heal" and teacher.seen == [] and trainer.calls == []


@pytest.mark.parametrize(
    ("heal", "gate", "match"),
    [
        ([TEST], DEV, "TEST task"),
        (TRAIN, [TEST], "TEST task"),
        (DEV, DEV, "heal tasks must be train"),
        (TRAIN, TRAIN, "gate tasks must be dev"),
        ([("telecom", "no-such-task")], DEV, "in no split"),
        ([TRAIN[0], TRAIN[0]], DEV, "listed twice in the heal tasks"),
        (TRAIN, [DEV[0], DEV[0]], "listed twice in the gate tasks"),
        (TRAIN, [], "no gate tasks"),
    ],
)
def test_splits_are_enforced_before_anything_runs(heal, gate, match):
    rollout = FakeRollout()
    agent, _, _ = make(rollout)
    with pytest.raises(ValueError, match=match):
        agent.run_cycle(heal, gate)
    assert rollout.calls == []


# --- Broken runs: an API error or a crash says nothing about the student ---------------------------


def broke(rec):
    return {**rec, "termination": "infrastructure_error", "error": "RateLimitError: too many requests"}


class BreakingRollout(FakeRollout):
    """FakeRollout, but the runs in ``breaks`` ((adapter, domain, task_id)) break."""

    def __init__(self, breaks, heals=frozenset()):
        super().__init__(heals)
        self.breaks = breaks

    def __call__(self, adapter, domain, task_id):
        rec = super().__call__(adapter, domain, task_id)
        return broke(rec) if (adapter, domain, task_id) in self.breaks else rec


def test_a_broken_heal_run_is_not_a_failure_and_never_reaches_the_teacher():
    rollout = BreakingRollout({(None, *TRAIN[0])}, heals=set(DEV))
    agent, teacher, trainer = make(rollout)
    report = agent.run_cycle(TRAIN, DEV)
    assert [r["task_id"] for r in report.broken] == [TRAIN[0][1]]
    assert [f.task for f in report.failures] == TRAIN[1:] and [f.task for f in teacher.seen] == TRAIN[1:]
    assert [fx["task_id"] for fx in trainer.calls[0][0]] == [t for _, t in TRAIN[1:]]
    assert report.summary()["broken"] == 1


def test_a_cycle_whose_heal_runs_all_broke_stops_before_the_teacher():
    agent, teacher, trainer = make(BreakingRollout({(None, *t) for t in TRAIN}))
    report = agent.run_cycle(TRAIN, DEV)
    assert report.skipped == "no failures to heal (3 broken runs left out)"
    assert teacher.seen == [] and trainer.calls == [] and agent.adapter is None


def test_a_gate_task_that_broke_for_either_adapter_is_left_out_of_both_sides():
    # The candidate passes every dev task, but its run of DEV[0] broke and the base's run of DEV[1] did
    rollout = BreakingRollout({("stub-adapter-1", *DEV[0]), (None, *DEV[1])}, heals=set(DEV))
    agent, _, _ = make(rollout)
    report = agent.run_cycle(TRAIN, DEV)
    assert report.gate_dropped == [DEV[0], DEV[1]]
    assert (report.gate.baseline_rate, report.gate.candidate_rate) == (0.0, 1.0)  # on DEV[2:] only
    assert len(report.gate_baseline) == len(report.gate_candidate) == len(DEV)  # every run is kept
    assert report.summary()["gate_dropped"] == [list(DEV[0]), list(DEV[1])]


def test_a_broken_candidate_run_doesnt_count_against_it():
    # Without pairing, the candidate's broken run would count as a fail: 2/4 = 0.5 against 0.5
    rollout = BreakingRollout({("stub-adapter-1", *DEV[0])}, heals=set(DEV[:2]))
    agent, _, _ = make(rollout)
    report = agent.run_cycle(TRAIN, DEV)
    assert report.gate.passed and report.gate.candidate_rate == pytest.approx(1 / 3)


def test_if_every_gate_task_broke_the_adapter_is_kept():
    agent, _, _ = make(BreakingRollout({("stub-adapter-1", *t) for t in DEV}, heals=set(DEV)))
    report = agent.run_cycle(TRAIN, DEV)
    assert not report.gate.passed and "every gate task broke" in report.gate.reason
    assert agent.adapter is None and report.gate_dropped == DEV


def test_the_report_keeps_the_gate_runs_and_its_summary_is_json():
    agent, _, _ = make(FakeRollout(heals=set(DEV[:2])))
    report = agent.run_cycle(TRAIN, DEV)
    assert [r["adapter"] for r in report.gate_baseline] == [None] * len(DEV)
    assert [r["adapter"] for r in report.gate_candidate] == ["stub-adapter-1"] * len(DEV)
    assert json.loads(json.dumps(report.summary()))["gate_dropped"] == []


def test_a_second_cycle_heals_from_the_adapter_the_first_one_kept():
    rollout = FakeRollout(heals=set(DEV[:2]))
    agent, _, trainer = make(rollout)
    agent.run_cycle(TRAIN, DEV)
    second = agent.run_cycle(TRAIN, DEV)
    assert trainer.calls[1][1] == "stub-adapter-1" and second.adapter_before == "stub-adapter-1"
    assert second.gate.baseline_rate == 0.5 and len(agent.history) == 2


def test_arm_rollout_runs_the_arm_with_the_agent_for_the_adapter(monkeypatch):
    calls = []

    def run_episode(arm, agent, domain, task_id, **kw):
        calls.append((arm, agent, domain, task_id, kw))
        return {"domain": domain, "task_id": task_id}

    monkeypatch.setitem(sys.modules, "reflexive.arms", types.SimpleNamespace(run_episode=run_episode))
    rollout = arm_rollout("arm-A", lambda adapter: f"agent for {adapter}", judges=False)
    rec = rollout("adapter-1", *DEV[0])
    assert calls == [("arm-A", "agent for adapter-1", *DEV[0], {"judges": False})]
    assert rec == {"domain": DEV[0][0], "task_id": DEV[0][1], "adapter": "adapter-1"}


def test_the_agreed_gate_runs_in_the_loop_and_reports_what_it_dropped():
    """DevGate on 4 dev tasks: the candidate's run of DEV[3] broke, so it's left out of both sides."""
    from test_gate import saved

    from reflexive.gate import DevGate

    rollout = BreakingRollout({("stub-adapter-1", *DEV[3])}, heals=set(DEV[:2]))
    gate = DevGate.load(min_tasks=len(DEV), score=saved)
    agent = ReflexiveAgent(rollout, teacher=StubTeacher(), trainer=StubTrainer(), gate=gate)
    report = agent.run_cycle(TRAIN, DEV)
    decision = report.gate
    assert decision is not None and decision.verdict == "promote" and agent.adapter == "stub-adapter-1"
    assert report.gate_dropped == [DEV[3]] and decision.details["net_gain"] == 2
    assert json.loads(json.dumps(report.summary()))["gate"]["check"] == "all"


def test_stop_after_fixes_trains_and_gates_nothing():
    rollout = FakeRollout()
    agent, teacher, trainer = make(rollout)
    report = agent.run_cycle(TRAIN, DEV, stop_after_fixes=True)
    assert len(report.fixes) == 3 and trainer.calls == [] and report.gate is None
    assert report.skipped.startswith("stopped after the fixes") and len(rollout.calls) == len(TRAIN)


def test_parallel_rollouts_keep_the_tasks_order():
    rollout = FakeRollout(heals=set(DEV[:2]))
    teacher, trainer = StubTeacher(), StubTrainer()
    agent = ReflexiveAgent(rollout, teacher=teacher, trainer=trainer, gate=PassRateGate(), workers=4)
    report = agent.run_cycle(TRAIN, DEV)
    assert [(r["domain"], r["task_id"]) for r in report.rollouts] == TRAIN
    assert [(r["domain"], r["task_id"]) for r in report.gate_candidate] == DEV and report.swapped
