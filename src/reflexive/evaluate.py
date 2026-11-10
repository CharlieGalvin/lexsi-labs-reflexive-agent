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

`summarise` turns a run's conversations into the arm's numbers: success with its 95% range,
the success rate of each sample, the scorers' means, TTC medians and means, and success per
million tokens and per dollar. Crashed conversations (`reflexive.metrics.is_broken`) are
counted, not scored. Works on any split, test included: it only reads results.

**Success:** only success and failure verdicts make a rate. An ``unknown`` (judges off) or an
``error`` (the oracle couldn't score it) could be either, so with any of them the summary gives
the range they span (``success_bounds``) and no rate. The 95% range resamples **tasks**, not
conversations: two samples of one task aren't independent, so counting them separately would
make the range too narrow.

**Per dollar** counts API spend only (the agent's and the customer's). A self-hosted student
(the pod arms) costs $0 there, so its successes per dollar can't be compared with an API arm's;
successes per million tokens can.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

from reflexive.files import latest_records
from reflexive.metrics import is_broken

Record = dict[str, Any]

SCORERS = ("tool_call_validity", "redundant_tool_calls", "agent_loop_detection")
JUDGED = "tool_selection"
REPS = 10_000
UNDECIDED = ("unknown", "error")  # verdicts that could be a success or a failure
COST_COVERS = "API calls only (agent and customer): a self-hosted student counts as $0"


@cache
def tool_schemas(domain: str) -> tuple[dict, ...]:
    """The domain's agent tools as OpenAI function schemas (what the agent was offered)."""
    from tau2.registry import registry

    env = registry.get_env_constructor(domain)()
    return tuple(t.openai_schema for t in env.get_tools() if t.name != "done")


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
        "task_id": f"{rec.get('domain')}:{rec.get('task_id')}",
        "sample": rec.get("sample", 0),
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


def task_bootstrap(rows: list[dict[str, Any]], reps: int = REPS, seed: int = 0) -> list[float] | None:
    """95% percentile-bootstrap range of the success rate, resampling tasks (all their samples)."""
    by_task: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if r["verdict"] in ("success", "failure"):
            by_task.setdefault(r.get("task_id") or f"row{i}", []).append(r["verdict"] == "success")
    tasks = list(by_task.values())
    if not tasks:
        return None
    rng = random.Random(seed)
    stats = []
    for _ in range(reps):
        pick = rng.choices(tasks, k=len(tasks))
        stats.append(sum(map(sum, pick)) / sum(map(len, pick)))
    stats.sort()
    return [round(stats[int(0.025 * reps)], 4), round(stats[min(int(0.975 * reps), reps - 1)], 4)]


def by_sample(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Success per sample (each sample is one full run over the tasks) and their mean."""
    rates = {}
    for s in sorted({r.get("sample", 0) for r in rows}):
        mine = [r for r in rows if r.get("sample", 0) == s]
        if any(r["verdict"] in UNDECIDED for r in mine):
            rates[str(s)] = None
            continue
        rates[str(s)] = round(sum(r["verdict"] == "success" for r in mine) / len(mine), 4) if mine else None
    known = [v for v in rates.values() if v is not None]
    mean = round(statistics.fmean(known), 4) if known and len(known) == len(rates) else None
    return {"rates": rates, "mean": mean}


def summarise(rows: list[dict[str, Any]], crashed: int = 0) -> dict[str, Any]:
    """An arm's headline numbers from its conversations' `evaluate_record` rows."""
    n_ok = sum(r["verdict"] == "success" for r in rows)
    n = sum(r["verdict"] in ("success", "failure") for r in rows)
    undecided = sum(r["verdict"] in UNDECIDED for r in rows)
    rated = bool(n) and not undecided  # an unknown or error could be either: bounds, not a rate
    out: dict[str, Any] = {
        "conversations": len(rows),
        "crashed_not_scored": crashed,
        "verdicts": {
            v: sum(r["verdict"] == v for r in rows) for v in ("success", "failure", "unknown", "error")
        },
        "success_rate": round(n_ok / n, 4) if rated else None,
        "success_ci95": task_bootstrap(rows) if rated else None,
        "ci_method": "percentile bootstrap over tasks (a task's samples together), 10,000 resamples",
        "success_by_sample": by_sample(rows),
        "success_bounds": [
            round(n_ok / (n + undecided), 4),
            round((n_ok + undecided) / (n + undecided), 4),
        ]
        if undecided
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
    out["cost_covers"] = COST_COVERS
    out["user_tokens_recorded"] = all(r.get("user_tokens") is not None for r in rows) if rows else None
    return out


def evaluate(
    records: list[Record], success: Callable[[Record], str], judge: Any = None, workers: int = 1
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(one row per scored conversation, the summary) for a run's records."""
    from concurrent.futures import ThreadPoolExecutor

    recs = latest_records(records)
    broken = [r for r in recs if is_broken(r)]
    ok = [r for r in recs if not is_broken(r)]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = list(pool.map(lambda r: evaluate_record(r, success, judge), ok))
    return rows, summarise(rows, crashed=len(broken))
