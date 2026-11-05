"""SEL-40: one ReflexiveAgent heal cycle end to end, with a fake rollout and the stub parts.

No GPU, no τ²-bench, no API: the rollout returns built records, and the classifier is the real one.
"""

from __future__ import annotations

import pytest

from reflexive.agent import PassRateGate, ReflexiveAgent, StubTeacher, StubTrainer
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
    ],
)
def test_splits_are_enforced_before_anything_runs(heal, gate, match):
    rollout = FakeRollout()
    agent, _, _ = make(rollout)
    with pytest.raises(ValueError, match=match):
        agent.run_cycle(heal, gate)
    assert rollout.calls == []
