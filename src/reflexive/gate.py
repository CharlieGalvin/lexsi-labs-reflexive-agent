"""The deployment gate (SEL-51): whether a healed adapter replaces the current one.

    gate = DevGate.load()                       # thresholds from configs/gate.yaml
    decision = gate.decide(current_runs, candidate_runs)
    decision.verdict                            # "promote" | "reject" | "no_decision"

Both adapters run the same dev tasks, and ``ReflexiveAgent`` re-runs the current adapter at every
gate. **Every run is re-scored by the AuditKIT oracle** (``reflexive.oracle.Tau2Oracle``, SEL-46:
τ²-bench's evaluator, judges on), never read from the record, so the gate decides on verified
state. The oracle's ``error`` (the run broke) is a crash; its ``unknown`` (a pass that skipped
judged checks) can't be decided on. The checks run in order, and the first that fails decides:

1. **Complete run**: at least ``min_tasks`` tasks, the same tasks for both adapters, one record
   per task, at most ``max_crashed`` crashed conversations each, and no ``unknown``. Every task
   must be a **dev** task: a train or test task is refused outright (gating on test would spend
   the final evaluation). Otherwise
   **no decision**: re-run the gate, don't reject. Tasks that crashed for either adapter are left
   out of both sides (``details["dropped"]``), so the comparison stays paired.
2. **Pass rate**: tasks only the candidate passed, minus tasks only the current adapter passed,
   is at least ``margin``.
3. **Hallucination**: findings per conversation (``classify_record``, of ``finding_types``) rise
   by at most ``findings_tolerance`` (relative), and always by at least one finding over the
   whole run. A conversation already has about a dozen findings (Arm A telecom dev: 11.7), so a
   strict "no rise" would reject on noise; the one-finding slack only matters near zero, where a
   relative tolerance would reject a single extra finding.
4. **Domains**: no domain loses ``domain_loss`` tasks or more (net, as in check 2).

Each decision also records the McNemar p-value and the paired bootstrap range of the gain, but
significance isn't required: on 36 tasks McNemar is underpowered, and the statistical claim comes
from the test split.

**Budget**: at most ``max_cycles`` promote-or-reject decisions on the same dev tasks; after that,
``decide`` gives no decision ("run the test split"), so the dev set isn't fitted to by repeated
gating. A no-decision doesn't use up a cycle. ``history`` keeps every decision, for the audit
record. Heal cycles run as separate jobs, days apart, so pass ``history_file`` (a JSON Lines
file) to keep the budget across them: every decision is appended to it, and ``load`` reads it
back, so a new gate knows how many cycles were already used.

The rule was agreed in SEL-51 by Chiara, Charlie and Katie (5-6 Nov). Its thresholds live in
``configs/gate.yaml``.
"""

from __future__ import annotations

import collections
import datetime as dt
import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from reflexive import splits
from reflexive.hallucination import FailureType, classify_record
from reflexive.stats import mcnemar, paired_bootstrap

Record = dict[str, Any]
Task = tuple[str, str]  # (domain, task_id)
Score = Callable[[Record], Any]  # a record -> an AuditKIT Outcome (.verdict, .reason)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "gate.yaml"
PROMOTE, REJECT, NO_DECISION = "promote", "reject", "no_decision"
SUCCESS, UNKNOWN, ERROR = "success", "unknown", "error"  # AuditKIT's verdicts (and "failure")


@dataclass(frozen=True)
class GateDecision:
    """What the gate decided, which check decided it, and the numbers behind it."""

    passed: bool  # promote the candidate
    baseline_rate: float
    candidate_rate: float
    reason: str = ""
    verdict: str = ""  # promote / reject / no_decision
    check: str = ""  # what decided: complete, pass_rate, hallucination, domains, budget, all
    details: dict[str, Any] = field(default_factory=dict)


def oracle_score(judges: bool = True) -> Score:
    """Score a record with the AuditKIT oracle; a record that isn't a conversation is an error."""
    from auditkit.agent_eval.outcome import Outcome

    from reflexive.oracle import Tau2Oracle, case_for, episode_from_record

    oracle = Tau2Oracle(judges=judges)

    def score(rec: Record) -> Any:
        try:
            ep = episode_from_record(rec)
        except Exception as e:  # noqa: BLE001 - one bad record is an error verdict, not a crash
            return Outcome(ERROR, reason=f"not a conversation ({type(e).__name__}: {e})")
        return oracle.evaluate(ep, case_for(rec, oracle))

    score.identity = oracle.identity()  # ty: ignore[unresolved-attribute]
    return score


def _task(rec: Record) -> Task:
    return (rec["domain"], str(rec["task_id"]))


@dataclass
class DevGate:
    """The agreed gate. Build it with ``DevGate.load()``; keyword arguments override the file.

    ``score`` defaults to the oracle with judges on (built on first use); tests pass a fake."""

    min_tasks: int
    max_crashed: int
    samples_per_task: int
    margin: int
    finding_types: tuple[str, ...]
    findings_tolerance: float
    domain_loss: int
    max_cycles: int
    bootstrap_reps: int = 10_000
    workers: int = 4  # conversations scored at once (the judges are API calls)
    score: Score | None = None
    history_file: Path | None = None  # JSON Lines: keeps the budget across separate runs
    history: list[GateDecision] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str | None = None, **override: Any) -> DevGate:
        cfg = yaml.safe_load(Path(path or CONFIG_PATH).read_text())
        settings = {f.name for f in fields(cls)} - {"history", "score", "workers", "history_file"}
        if unknown := set(cfg) - settings:
            raise ValueError(f"unknown gate settings in {path or CONFIG_PATH}: {', '.join(sorted(unknown))}")
        if unknown := set(override) - settings - {"score", "workers", "history_file"}:
            raise ValueError(f"unknown gate settings passed to DevGate.load: {', '.join(sorted(unknown))}")
        gate = cls(**{**cfg, **override})
        gate.finding_types = tuple(FailureType(t).value for t in gate.finding_types)  # a typo fails here
        if gate.history_file is not None:
            gate.history_file = Path(gate.history_file)
            gate.history = read_history(gate.history_file)
        return gate

    @property
    def cycles_used(self) -> int:
        """Promote-or-reject decisions so far: each one looked at the dev results."""
        return sum(d.verdict in (PROMOTE, REJECT) for d in self.history)

    def decide(self, baseline: Sequence[Record], candidate: Sequence[Record]) -> GateDecision:
        """Promote, reject or no decision, for the candidate's runs against the current one's."""
        decision = self._decide(list(baseline), list(candidate))
        self.history.append(decision)
        if self.history_file is not None:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            with self.history_file.open("a") as f:
                f.write(json.dumps(_to_json(decision)) + "\n")
        return decision

    # --- The checks -------------------------------------------------------------------------------

    def _decide(self, baseline: list[Record], candidate: list[Record]) -> GateDecision:
        def out(
            verdict: str,
            check: str,
            reason: str,
            details: dict[str, Any],
            rates: tuple[float, float] = (0.0, 0.0),
        ) -> GateDecision:
            return GateDecision(verdict == PROMOTE, *rates, reason, verdict, check, details)

        if self.cycles_used >= self.max_cycles:
            why = (
                f"{self.cycles_used} heal cycles already gated on these dev tasks "
                f"(max_cycles {self.max_cycles}): run the test split"
            )
            return out(NO_DECISION, "budget", why, {})
        if problem := self._shape(baseline, candidate):
            return out(NO_DECISION, "complete", f"{problem}: re-run the gate", {})

        if self.score is None:
            self.score = oracle_score()  # one oracle for the gate's lifetime
        meta = {"scored_by": getattr(self.score, "identity", None)}
        b, c = {_task(r): r for r in baseline}, {_task(r): r for r in candidate}
        vb, vc = self._verdicts(b), self._verdicts(c)
        for name, v in (("current adapter", vb), ("candidate", vc)):
            crashed = sorted(t for t, o in v.items() if o.verdict == ERROR)
            if len(crashed) > self.max_crashed:
                why = f"{len(crashed)} of the {name}'s conversations crashed (at most {self.max_crashed})"
                details = {**meta, "crashed": [list(t) for t in crashed]}
                return out(NO_DECISION, "complete", f"{why}: re-run the gate", details)
            if unknown := sorted(t for t, o in v.items() if o.verdict == UNKNOWN):
                why = f"{len(unknown)} of the {name}'s conversations are unverified ({v[unknown[0]].reason})"
                return out(NO_DECISION, "complete", f"{why}: re-run the gate", meta)

        dropped = sorted(t for t in b if ERROR in (vb[t].verdict, vc[t].verdict))
        scored = sorted(t for t in b if t not in dropped)
        b_pass = {t: vb[t].verdict == SUCCESS for t in scored}
        c_pass = {t: vc[t].verdict == SUCCESS for t in scored}
        n = len(scored) or 1
        rates = (sum(b_pass.values()) / n, sum(c_pass.values()) / n)
        only_b, only_c, p = mcnemar(b_pass, c_pass)
        lo, hi = paired_bootstrap(c_pass, b_pass, reps=self.bootstrap_reps) if scored else (0.0, 0.0)
        findings = {"baseline": self._findings(b, scored), "candidate": self._findings(c, scored)}
        domains = self._domains(scored, b_pass, c_pass)
        gain = only_c - only_b
        details = {
            **meta,
            "tasks": len(b),
            "scored": len(scored),
            "dropped": [list(t) for t in dropped],
            "passed": {"baseline": sum(b_pass.values()), "candidate": sum(c_pass.values())},
            "only_baseline_passed": only_b,
            "only_candidate_passed": only_c,
            "net_gain": gain,
            "mcnemar_p": round(p, 4),
            "bootstrap95_gain": [round(lo, 4), round(hi, 4)],
            "findings_per_conversation": findings,
            "domains_net": domains,
        }

        if gain < self.margin:
            why = (
                f"net gain {gain:+d} tasks ({only_c} only the candidate passed, {only_b} only the "
                f"current adapter passed); needs {self.margin:+d}"
            )
            return out(REJECT, "pass_rate", why, details, rates)
        fb, fc = findings["baseline"]["total"], findings["candidate"]["total"]
        allowed = round(max(fb * (1 + self.findings_tolerance), fb + 1 / n), 4)  # >= 1 finding of slack
        if fc > allowed:
            why = (
                f"{'+'.join(self.finding_types)} findings per conversation rose {fb:.2f} -> {fc:.2f}, "
                f"more than the {self.findings_tolerance:.0%} tolerance (up to {allowed:.2f})"
            )
            return out(REJECT, "hallucination", why, details, rates)
        if lost := {d: k for d, k in domains.items() if k <= -self.domain_loss}:
            worst = ", ".join(f"{d} {k:+d}" for d, k in sorted(lost.items()))
            return out(REJECT, "domains", f"a domain lost {self.domain_loss}+ tasks: {worst}", details, rates)
        why = (
            f"net gain {gain:+d} tasks (needs {self.margin:+d}); findings per conversation "
            f"{fb:.2f} -> {fc:.2f} (up to {allowed:.2f}); no domain lost {self.domain_loss}+ "
            f"(McNemar p {p:.3f}, not required)"
        )
        return out(PROMOTE, "all", why, details, rates)

    def _shape(self, baseline: list[Record], candidate: list[Record]) -> str | None:
        """Why the runs can't be compared at all (checked before scoring), or None.

        Raises on what is a mistake rather than a bad run: a task that isn't dev, or a task twice."""
        for name, recs in (("current adapter", baseline), ("candidate", candidate)):
            for t in sorted(set(map(_task, recs))):
                try:
                    splits.assert_not_test(*t)  # the test split is never gated on
                    split = splits.split_of(*t)
                except KeyError as e:
                    raise ValueError(f"{name}: {t[0]} task {t[1]!r} is in no split") from e
                if split != "dev":
                    raise ValueError(
                        f"{name}: {t[0]} task {t[1]!r} is a {split} task; the gate takes dev tasks only"
                    )
            per_task = collections.Counter(map(_task, recs))
            if dup := [t for t, k in per_task.items() if k > self.samples_per_task]:
                raise ValueError(
                    f"{name}: {dup[0][0]} task {dup[0][1]!r} has {per_task[dup[0]]} records; "
                    f"the gate takes {self.samples_per_task} per task"
                )
            if len(per_task) < self.min_tasks:
                return f"the {name} ran {len(per_task)} tasks, fewer than {self.min_tasks}"
        if (bt := set(map(_task, baseline))) != (ct := set(map(_task, candidate))):
            diff = sorted(bt ^ ct)
            return f"the adapters ran different tasks ({len(diff)} differ, e.g. {diff[0][0]} {diff[0][1]})"
        return None

    def _verdicts(self, recs: dict[Task, Record]) -> dict[Task, Any]:
        score, tasks = self.score, list(recs)
        assert score is not None
        with ThreadPoolExecutor(max(1, self.workers)) as pool:
            return dict(zip(tasks, pool.map(lambda t: score(recs[t]), tasks), strict=True))

    def _findings(self, recs: dict[Task, Record], scored: list[Task]) -> dict[str, float]:
        """Findings per scored conversation, by type and in total (``finding_types`` only)."""
        counts: collections.Counter[str] = collections.Counter()
        for t in scored:
            counts.update(f.type.value for f in classify_record(recs[t]).findings)
        n = len(scored) or 1
        per = {k: round(counts[k] / n, 4) for k in self.finding_types}
        return {**per, "total": round(sum(counts[k] for k in self.finding_types) / n, 4)}

    @staticmethod
    def _domains(scored: list[Task], b: dict[Task, bool], c: dict[Task, bool]) -> dict[str, int]:
        """Per domain: tasks only the candidate passed, minus tasks only the current one passed."""
        net: collections.Counter[str] = collections.Counter()
        for t in scored:
            net[t[0]] += int(c[t]) - int(b[t])
        return {d: net[d] for d in sorted({t[0] for t in scored})}


def _to_json(d: GateDecision) -> dict[str, Any]:
    return {"decided_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **vars(d)}


def read_history(path: Path) -> list[GateDecision]:
    """The decisions saved in a history file (none if it doesn't exist yet)."""
    if not path.exists():
        return []
    names = {f.name for f in fields(GateDecision)}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [GateDecision(**{k: v for k, v in r.items() if k in names}) for r in rows]
