"""The self-healing loop's shape (SEL-40, part of SEL-14): one heal cycle, end to end.

    agent = ReflexiveAgent(rollout, teacher=StubTeacher(), trainer=StubTrainer(), gate=PassRateGate())
    report = agent.run_cycle(heal_tasks, gate_tasks)

One cycle:

1. **Rollout**: the student, with its current adapter, on every heal task.
2. **Classify**: each failed conversation through `classify_record` (Tool / Arg / Loop).
3. **Fix**: the teacher turns a failure into a corrected trajectory, or gives up (``None``).
4. **Retrain**: the trainer makes a candidate adapter from the fixes.
5. **Gate**: the current and the candidate adapter both run the gate tasks; the gate decides.
6. **Swap**: the agent keeps the candidate only if the gate passes.

Classify is the real classifier. Rollout is whatever callable it's given: `arm_rollout` wraps
`reflexive.arms.run_episode`, and the tests pass a fake. Fix, retrain and gate are stubs for now,
behind small protocols, so the real ones drop in without changing the loop:

- teacher: SEL-47 (the real GLM-5.3-Flash fix run)
- trainer: the Trainer's AgentTune GRPO / BCO step
- gate: SEL-51 (the agreed deployment rule)

Heal tasks come from **train**, gate tasks from **dev**; a test task anywhere is refused
(`reflexive.splits.assert_not_test`). Nothing here needs a GPU, τ²-bench or an API key.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from reflexive.hallucination import Diagnosis, FailureType, classify_record
from reflexive.splits import assert_not_test, split_of

Record = dict[str, Any]  # a `reagent.tau2.episode` record
Task = tuple[str, str]  # (domain, task_id)
Adapter = str | None  # an adapter id; None is the base student
Rollout = Callable[[Adapter, str, str], Record]  # (adapter, domain, task_id) -> record


def passed(rec: Record) -> bool:
    return bool((rec.get("outcome") or {}).get("passed"))


def pass_rate(records: Sequence[Record]) -> float:
    return sum(map(passed, records)) / len(records) if records else 0.0


# --- The parts other roles are building ---------------------------------------------------------


@dataclass
class Failure:
    """A failed heal-task conversation and what the classifier found in it."""

    record: Record
    diagnosis: Diagnosis

    @property
    def task(self) -> Task:
        return (self.record["domain"], str(self.record["task_id"]))


class Teacher(Protocol):
    def fix(self, failure: Failure) -> Record | None:
        """A corrected trajectory for this failure, or None if the teacher couldn't fix it."""


class Trainer(Protocol):
    def train(self, fixes: Sequence[Record], base: Adapter) -> Adapter:
        """A new adapter trained from ``base`` on ``fixes``."""


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    baseline_rate: float
    candidate_rate: float
    reason: str = ""


class Gate(Protocol):
    def decide(self, baseline: Sequence[Record], candidate: Sequence[Record]) -> GateDecision:
        """Whether the candidate adapter should replace the current one."""


@dataclass
class StubTeacher:
    """Stands in for the teacher fix run (SEL-47): 'fixes' every failure it's not told to skip."""

    give_up_on: set[Task] = field(default_factory=set)
    seen: list[Failure] = field(default_factory=list)

    def fix(self, failure: Failure) -> Record | None:
        self.seen.append(failure)
        if failure.task in self.give_up_on:
            return None
        domain, task_id = failure.task
        return {
            "format": "stub.fix",
            "domain": domain,
            "task_id": task_id,
            "fixes": failure.diagnosis.failure,
        }


@dataclass
class StubTrainer:
    """Stands in for the GRPO / BCO step: names a new adapter and keeps what it was given."""

    calls: list[tuple[list[Record], Adapter]] = field(default_factory=list)

    def train(self, fixes: Sequence[Record], base: Adapter) -> Adapter:
        self.calls.append((list(fixes), base))
        return f"stub-adapter-{len(self.calls)}"


@dataclass(frozen=True)
class PassRateGate:
    """Placeholder rule until SEL-51: promote if the candidate's dev pass rate beats the baseline's
    by more than ``min_gain``."""

    min_gain: float = 0.0

    def decide(self, baseline: Sequence[Record], candidate: Sequence[Record]) -> GateDecision:
        b, c = pass_rate(baseline), pass_rate(candidate)
        ok = c - b > self.min_gain
        return GateDecision(
            ok, b, c, f"dev pass rate {b:.2f} -> {c:.2f}, needs a gain over {self.min_gain:.2f}"
        )


# --- The loop -------------------------------------------------------------------------------


@dataclass
class CycleReport:
    """Everything one cycle did, for the audit record (SEL-47) and the demo."""

    adapter_before: Adapter
    adapter_after: Adapter
    rollouts: list[Record]
    failures: list[Failure]
    fixes: list[Record]
    candidate: Adapter
    gate: GateDecision | None
    skipped: str = ""  # why the cycle stopped early, if it did

    @property
    def swapped(self) -> bool:
        return self.adapter_after != self.adapter_before

    def summary(self) -> dict[str, Any]:
        return {
            "adapter_before": self.adapter_before,
            "adapter_after": self.adapter_after,
            "swapped": self.swapped,
            "rollouts": len(self.rollouts),
            "passed": sum(map(passed, self.rollouts)),
            "failures": {t.value: sum(f.diagnosis.failure is t for f in self.failures) for t in FailureType},
            "fixes": len(self.fixes),
            "candidate": self.candidate,
            "gate": None if self.gate is None else vars(self.gate),
            "skipped": self.skipped,
        }


class ReflexiveAgent:
    """The student plus its heal loop. ``adapter`` is the one it serves with now."""

    def __init__(
        self,
        rollout: Rollout,
        *,
        teacher: Teacher,
        trainer: Trainer,
        gate: Gate,
        adapter: Adapter = None,
    ) -> None:
        self.rollout = rollout
        self.teacher = teacher
        self.trainer = trainer
        self.gate = gate
        self.adapter = adapter
        self.history: list[CycleReport] = []

    def _run(self, adapter: Adapter, tasks: Sequence[Task]) -> list[Record]:
        return [self.rollout(adapter, d, t) for d, t in tasks]

    def run_cycle(self, heal_tasks: Sequence[Task], gate_tasks: Sequence[Task]) -> CycleReport:
        """Rollout → classify → fix → retrain → gate → swap. Returns what happened."""
        _check(heal_tasks, "train", "heal")
        _check(gate_tasks, "dev", "gate")
        before = self.adapter

        rollouts = self._run(before, heal_tasks)
        failures = [Failure(r, classify_record(r)) for r in rollouts if not passed(r)]
        fixes = [fx for f in failures if (fx := self.teacher.fix(f)) is not None]

        def done(
            candidate: Adapter = None, gate: GateDecision | None = None, skipped: str = ""
        ) -> CycleReport:
            report = CycleReport(before, self.adapter, rollouts, failures, fixes, candidate, gate, skipped)
            self.history.append(report)
            return report

        if not failures:
            return done(skipped="no failures to heal")
        if not fixes:
            return done(skipped="the teacher fixed none of the failures")

        candidate = self.trainer.train(fixes, before)
        decision = self.gate.decide(self._run(before, gate_tasks), self._run(candidate, gate_tasks))
        if decision.passed:
            self.adapter = candidate
        return done(candidate, decision)


def _check(tasks: Sequence[Task], want: str, role: str) -> None:
    """Never touch test; heal only on train and gate only on dev, so the gate isn't graded on
    what was trained on."""
    for domain, task_id in tasks:
        assert_not_test(domain, task_id)
        if (got := split_of(domain, task_id)) != want:
            raise ValueError(f"{domain} task {task_id!r} is a {got} task; {role} tasks must be {want}")


def arm_rollout(arm: Any, make_agent: Callable[[Adapter], Any], **kwargs: Any) -> Rollout:
    """A real `Rollout`: `reflexive.arms.run_episode` with the agent ``make_agent(adapter)`` builds.

    The student behind Cohere's API can't load an adapter, so for now ``make_agent`` should
    refuse anything but None; the GPU-served student will take the adapter id.
    """
    from reflexive.arms import run_episode

    def rollout(adapter: Adapter, domain: str, task_id: str) -> Record:
        rec = run_episode(arm, make_agent(adapter), domain, task_id, **kwargs)
        rec["adapter"] = adapter
        return rec

    return rollout
