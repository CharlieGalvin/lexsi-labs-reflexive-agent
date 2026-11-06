"""τ²-bench's own scoring as an AuditKIT outcome oracle (SEL-46).

    from reflexive.oracle import Tau2Oracle, case_for, episode_from_record

    oracle = Tau2Oracle(judges=True)
    outcome = oracle.evaluate(episode_from_record(rec), case_for(rec, oracle))
    outcome.verdict          # "success" | "failure" | "unknown" | "error"

AuditKIT's built-in oracles check a final answer's text or a known end state; they can't tell
whether a τ²-bench task was really done. This oracle re-scores the conversation with
``reflexive.outcome.score`` (τ²-bench's evaluator: the databases' final state, plus the
communicate and natural-language checks), so every success rate the team reports, and the
deployment gate, come from verified state.

**Verdicts.**

- ``success`` / ``failure``: τ²-bench passed or failed it.
- ``error``: the run broke (a crash, a rate limit: ``reflexive.runner.is_broken``) or the
  scorer raised. The model wasn't tested, so it's never a ``failure``.
- ``unknown``: with ``judges=False``, a pass that skipped LLM-judged checks
  (``Outcome.verified`` is false). That's a best case, not a verified pass (AuditKIT's AG-07:
  missing evidence is never turned into a pass), including a task whose checks are all
  judged, where nothing ran at all. A failed deterministic check, or a conversation that
  ended early (τ²-bench scores it 0 without running any check), is a definite ``failure``.
  A record with no ending is an ``error``.

**Re-scored, not read.** The record's own saved ``outcome`` is passed on only as
``episode.source_verdict``: AuditKIT never uses it as the result (its AG-05) but reports any
disagreement with the oracle, which makes it a free consistency check on our runs. A saved
pass that skipped judged checks isn't compared; a saved failure always is.

**What the episode carries.** Its events are what the agent saw (``agent_view_of``). The
oracle also needs the customer's side, so the fields it scores from (``SCORED_FIELDS``) ride on
the episode as ``tau2_record``, an attribute ``to_dict`` doesn't serialise: AuditKIT reports
neither store each conversation twice nor show the customer's own actions.
"""

from __future__ import annotations

from typing import Any

from auditkit.agent_eval.importers import episode_from_openai_messages
from auditkit.agent_eval.outcome import ERROR, FAILURE, SUCCESS, UNKNOWN, Oracle, Outcome
from auditkit.agent_eval.types import AgentCase, AgentEpisode

from reflexive.hallucination import agent_view_of
from reflexive.runner import is_broken

Record = dict[str, Any]

TAU2_VERSION = "2.0.0"  # third_party/tau2-bench, pinned by `just tau2`
SCORING_VERSION = "3"  # bump when the scoring or the verdict rules change (v3, 6 Nov: Outcome.verified)
SCORED_FIELDS = ("domain", "task_id", "messages", "termination", "error")


def record_of(episode: Any) -> Record:
    """The fields the oracle scores from, as ``episode_from_record`` attached them."""
    rec = getattr(episode, "tau2_record", None)
    if not isinstance(rec, dict):
        raise ValueError("episode has no τ²-bench record: build it with episode_from_record()")
    return rec


class Tau2Oracle(Oracle):
    """Pass or fail from τ²-bench's evaluator, with the checks behind it as evidence.

    An AuditKIT ``Oracle`` with its own ``evaluate``: a ``CustomPredicate`` would only carry a
    bare verdict, without the reason and evidence."""

    type = "tau2_score"

    def __init__(self, *, judges: bool = True, language: str = "en") -> None:
        self.judges, self.language = judges, language
        self.version = (
            f"tau2-bench {TAU2_VERSION}; scoring v{SCORING_VERSION}; judges {'on' if judges else 'off'}"
        )
        self.name = "reflexive.outcome.score"

    def identity(self) -> dict[str, Any]:
        return {"type": self.type, "version": self.version, "fn": self.name}

    def evaluate(self, episode: Any, case: Any = None) -> Outcome:
        try:
            rec = record_of(episode)
            if is_broken(rec):  # first: a crash with no ending keeps its error text
                return self._out(
                    ERROR,
                    f"the run broke ({rec.get('error') or rec.get('termination')}): "
                    "not scored, and not a model failure",
                    {"error": (rec.get("error") or "")[:300], "termination": rec.get("termination")},
                )
            if not rec.get("termination"):
                return self._out(ERROR, "the record has no ending (termination): it can't be scored", {})
            from reflexive.outcome import score

            o = score(
                rec["domain"],
                str(rec["task_id"]),
                rec["messages"],
                rec["termination"],
                judges=self.judges,
                language=self.language,
            )
        except Exception as e:  # a broken scorer is a method error, never a model failure
            return self._out(ERROR, f"scoring raised {type(e).__name__}: {str(e)[:200]}", {})
        evidence = {
            "reward": o.reward,
            "checks": o.checks,
            "judged": o.judged,
            "skipped": o.skipped,
            "note": o.note,
            "termination": rec.get("termination"),
        }
        if not o.verified:  # a pass with judged checks skipped: a best case, not a verified pass
            ran = f"deterministic checks ({', '.join(o.checks)}) passed" if o.checks else "no check ran"
            return self._out(
                UNKNOWN, f"{ran}, but {', '.join(o.skipped)} were skipped (judges off)", evidence
            )
        if o.passed:
            return self._out(SUCCESS, "τ²-bench passed it (reward 1)", evidence)
        failed = [k for k, v in o.checks.items() if v < 1.0]
        why = o.note or (f"failed {', '.join(failed)}" if failed else f"reward {o.reward}")
        return self._out(FAILURE, f"τ²-bench failed it: {why}", evidence)

    def _out(self, verdict: str, reason: str, evidence: dict[str, Any]) -> Outcome:
        return Outcome(verdict, reason=reason, evidence=evidence, source=self.identity())


def episode_from_record(rec: Record) -> AgentEpisode:
    """One rollout record as an AuditKIT episode.

    The events are what the agent saw and did (``agent_view_of``: without the customer's own
    phone actions), for AuditKIT's agent scorers. The fields the oracle scores from ride along
    as ``episode.tau2_record`` (not serialised: see the module docstring). The record's saved
    verdict is ``source_verdict``: provenance only. ``KeyError`` without a domain or task_id.
    """
    saved_outcome = rec.get("outcome") or {}
    saved = saved_outcome.get("passed")
    # A pass saved with judges off isn't verified, so it can't be compared; a failure is definite
    if saved and saved_outcome.get("skipped"):
        saved = None
    view = [m for m in agent_view_of(rec) if m.get("role") != "system"]
    ep = episode_from_openai_messages(
        view,
        case_id=f"{rec['domain']}:{rec['task_id']}",
        source_id=f"{rec.get('arm', '?')}-{rec['domain']}-{rec['task_id']}-{rec.get('sample', 0)}",
        source="reagent.tau2.episode/v0",
    )
    # AuditKIT books a run as the target's fault (not the scorer's) when stop_reason is "error"
    ep.stop_reason = "error" if is_broken(rec) else rec.get("termination")
    ep.errors = [rec["error"]] if rec.get("error") else []
    ep.source_reward = (rec.get("outcome") or {}).get("reward")
    ep.source_verdict = None if is_broken(rec) or saved is None else (SUCCESS if saved else FAILURE)
    ep.tau2_record = {k: rec.get(k) for k in SCORED_FIELDS}  # ty: ignore[unresolved-attribute]
    return ep


def case_for(rec: Record, oracle: Tau2Oracle) -> AgentCase:
    """The AuditKIT case for a record's τ²-bench task, judged by ``oracle``."""
    return AgentCase(
        id=f"{rec['domain']}:{rec['task_id']}",
        task=f"τ²-bench {rec['domain']} task {rec['task_id']}",
        category=f"tau2/{rec['domain']}",
        outcome=oracle,
        metadata={"domain": rec["domain"], "task_id": str(rec["task_id"])},
    )
