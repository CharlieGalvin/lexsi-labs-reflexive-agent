"""The headline metrics for one arm on τ²-bench (SEL-38): success, AuditKIT's agent scorers, TTC.

Per conversation (`evaluate_record`):

- **success**: the AuditKIT oracle's verdict (`reflexive.oracle`, τ²-bench's own scoring,
  re-scored, never read from the record), or the saved verdict with ``success="saved"``.
- **AuditKIT agent scorers** (`auditkit.metrics.agent`), on the agent's side of the conversation:
  ``tool_call_validity`` (share of calls valid against the domain's tool schemas),
  ``redundant_tool_calls`` (share of calls repeating an earlier identical call),
  ``agent_loop_detection`` (1 = no loop: repeated calls, near-identical consecutive replies or a
  call cycle) and, optionally, ``tool_selection`` (an LLM judge: was each call justified?).
- **TTC**, time and tokens to complete: the agent's prompt and completion tokens, the simulated
  customer's tokens (when the record has them; records from before 8 Nov have only its cost),
  the cost of both, and the wall-clock seconds.

`summarise` turns a run's conversations into the arm's numbers: success with its 95% Wilson
range, the scorers' means, TTC medians and means, and success per million tokens and per
dollar. Crashed conversations (`reflexive.metrics.is_broken`) are counted, not scored. Works on
any split, test included: it only reads results.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

from reflexive.metrics import is_broken
from reflexive.stats import wilson

Record = dict[str, Any]

SCORERS = ("tool_call_validity", "redundant_tool_calls", "agent_loop_detection")
JUDGED = "tool_selection"


@cache
def tool_schemas(domain: str) -> tuple[dict, ...]:
    """The domain's agent tools as OpenAI function schemas (what the agent was offered)."""
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)()
    return tuple(t.openai_schema for t in env.get_tools() if t.name != "done")


def latest(records: Iterable[Record]) -> list[Record]:
    """The last record per (task, sample): a re-run of a crashed conversation replaces it."""
    keep: dict[tuple, Record] = {}
    for r in records:
        keep[(r.get("domain"), str(r.get("task_id")), r.get("sample", 0))] = r
    return list(keep.values())


def agent_scores(rec: Record, schemas: Iterable[dict], judge: Any = None) -> dict[str, Any]:
    """AuditKIT's agent scorers on one conversation: {scorer: value or None, ...}.

    None when a scorer has nothing to measure (no tool calls). ``judge``: an AuditKIT
    ``ToolSelectionJudge`` for ``tool_selection``, or None to skip it."""
    from auditkit.metrics.agent import AgentLoopDetection, RedundantToolCalls, ToolCallValidity
    from auditkit.sample import Sample

    msgs = rec.get("messages") or []
    first = next((m.get("content") for m in msgs if m.get("role") == "user" and m.get("content")), "")
    sample = Sample(input=first or str(rec.get("task_id")), tools=list(schemas))
    context = {"trace": {"messages": msgs}}
    out: dict[str, Any] = {}
    loop_why = None
    metrics = [ToolCallValidity(), RedundantToolCalls(), AgentLoopDetection()]
    if judge is not None:
        metrics.append(judge)
    for m in metrics:
        got = m.score(sample, "", context)
        got = got if isinstance(got, list) else [got]
        out[m.name] = round(got[0].value, 4) if got and got[0].value is not None else None
        if m.name == "agent_loop_detection" and got:
            loop_why = got[0].reason
    out["loop_signals"] = loop_why
    return out


def ttc(rec: Record) -> dict[str, Any]:
    """Time and tokens to complete one conversation, agent and simulated customer."""
    au = rec.get("agent_usage") or {}
    uu = rec.get("user_usage") or {}
    agent_tokens = (au.get("prompt_tokens") or 0) + (au.get("completion_tokens") or 0)
    user_tokens = (uu.get("prompt_tokens") or 0) + (uu.get("completion_tokens") or 0) if uu else None
    agent_cost, user_cost = au.get("est_cost"), rec.get("user_cost")
    costs = [c for c in (agent_cost, user_cost) if c is not None]
    return {
        "agent_tokens": agent_tokens,
        "agent_calls": au.get("calls"),
        "user_tokens": user_tokens,
        "total_tokens": agent_tokens + (user_tokens or 0) if user_tokens is not None else None,
        "cost_usd": round(sum(costs), 5) if costs else None,
        "seconds": rec.get("wall_s") or rec.get("duration_s"),
        "steps": sum(1 for m in rec.get("messages") or [] if m.get("role") == "assistant"),
    }


def evaluate_record(rec: Record, success: Callable[[Record], str], judge: Any = None) -> dict[str, Any]:
    """Everything SEL-38 reports for one conversation."""
    return {
        "task": f"{rec.get('domain')}:{rec.get('task_id')} #{rec.get('sample', 0)}",
        "domain": rec.get("domain"),
        "verdict": success(rec),
        "termination": rec.get("termination"),
        **agent_scores(rec, tool_schemas(rec["domain"]), judge),
        **ttc(rec),
    }


def saved_verdict(rec: Record) -> str:
    return "success" if (rec.get("outcome") or {}).get("passed") else "failure"


def oracle_verdict(oracle: Any) -> Callable[[Record], str]:
    """The AuditKIT oracle's verdict for a record (success / failure / unknown / error)."""

    def verdict(rec: Record) -> str:
        from auditkit.agent_eval.outcome import resolve_outcome

        from reflexive.oracle import case_for, episode_from_record

        try:
            return resolve_outcome(episode_from_record(rec), case_for(rec, oracle)).verdict
        except Exception:  # noqa: BLE001 - an unreadable record is an error verdict, not a crash
            return "error"

    return verdict


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 4) if xs else None


def _median(xs: list[float]) -> float | None:
    return round(statistics.median(xs), 2) if xs else None


def summarise(rows: list[dict[str, Any]], crashed: int = 0) -> dict[str, Any]:
    """An arm's headline numbers from its conversations' `evaluate_record` rows."""
    n_ok = sum(r["verdict"] == "success" for r in rows)
    n = sum(r["verdict"] in ("success", "failure") for r in rows)
    unknown = sum(r["verdict"] == "unknown" for r in rows)
    lo, hi = wilson(n_ok, n)
    rated = bool(n) and not unknown  # an unknown could be either: give bounds, not a rate
    out: dict[str, Any] = {
        "conversations": len(rows),
        "crashed_not_scored": crashed,
        "verdicts": {
            v: sum(r["verdict"] == v for r in rows) for v in ("success", "failure", "unknown", "error")
        },
        "success_rate": round(n_ok / n, 4) if rated else None,
        "success_wilson95": [round(lo, 4), round(hi, 4)] if rated else None,
        "success_bounds": [round(n_ok / (n + unknown), 4), round((n_ok + unknown) / (n + unknown), 4)]
        if unknown
        else None,
    }
    for s in (*SCORERS, JUDGED):
        vals = [r[s] for r in rows if r.get(s) is not None]
        if vals or s != JUDGED:
            out[s] = _mean(vals)
    out["conversations_with_invalid_call"] = sum(
        1 for r in rows if r.get("tool_call_validity") is not None and r["tool_call_validity"] < 1
    )
    out["conversations_looping"] = sum(1 for r in rows if r.get("agent_loop_detection") == 0)
    for k in ("agent_tokens", "user_tokens", "total_tokens", "cost_usd", "seconds", "steps"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[f"{k}_median"] = _median(vals)
        out[f"{k}_mean"] = round(statistics.fmean(vals), 2) if vals else None
    tokens = sum(r["agent_tokens"] + (r.get("user_tokens") or 0) for r in rows)
    cost = sum(r["cost_usd"] or 0 for r in rows)
    out["successes_per_million_tokens"] = round(n_ok / tokens * 1e6, 3) if tokens else None
    out["successes_per_dollar"] = round(n_ok / cost, 3) if cost else None
    out["user_tokens_recorded"] = all(r.get("user_tokens") is not None for r in rows) if rows else None
    return out


def evaluate(
    records: list[Record], success: Callable[[Record], str], judge: Any = None, workers: int = 1
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(one row per scored conversation, the summary) for a run's records."""
    from concurrent.futures import ThreadPoolExecutor

    recs = latest(records)
    broken = [r for r in recs if is_broken(r)]
    ok = [r for r in recs if not is_broken(r)]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = list(pool.map(lambda r: evaluate_record(r, success, judge), ok))
    return rows, summarise(rows, crashed=len(broken))
