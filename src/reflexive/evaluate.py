"""The headline metrics for one arm on τ²-bench (SEL-38): success, AuditKIT's agent scorers, TTC.

Per conversation (`evaluate_record`):

- **success**: the AuditKIT oracle's verdict (`reflexive.oracle`, τ²-bench's own scoring,
  re-scored, never read from the record), or the saved verdict with ``success="saved"``.
- **AuditKIT agent scorers** (`auditkit.metrics.agent`), on the conversation as the agent saw it
  (`agent_view_of`: the customer's own phone actions are left out, they aren't the agent's), with
  the domain's tools from ``configs/tool_schemas.json`` (no τ²-bench needed):
  ``tool_call_validity`` (share of calls valid against the domain's tool schemas),
  ``redundant_tool_calls`` (share of calls repeating an earlier identical call),
  ``agent_loop_detection`` (1 = no loop: repeated calls, near-identical consecutive replies or a
  call cycle) and, optionally, ``tool_selection`` (an LLM judge: was each call justified?).
- **TTC**, time and tokens to complete: the agent's prompt and completion tokens, the simulated
  customer's tokens (when the record has them; records from before 8 Nov have only its cost),
  the cost of both (`reflexive.metrics.record_cost`), and the wall-clock seconds.

`summarise` turns a run's conversations into the arm's numbers: success with its 95% range,
the success rate of each sample, the scorers' means, TTC medians and means, and success per
million tokens and per dollar. Crashed conversations (`reflexive.metrics.is_broken`) are
counted, not scored, but their spend is in the cost (as in `reflexive.metrics.arm_metrics`).
Works on any split, test included: it only reads results.

**Success:** only success and failure verdicts make a rate. An ``unknown`` (judges off) or an
``error`` (the oracle couldn't score it) could be either, so with any of them the summary gives
the range they span (``success_bounds``) and no rate. The 95% range resamples **tasks**, not
conversations: two samples of one task aren't independent, so counting them separately would
make the range too narrow (`reflexive.stats.cluster_bootstrap`). A pass saved with judges off
(``outcome.skipped``) is ``unknown`` under ``success="saved"`` too, as the oracle has it.

**Efficiency** (successes per million tokens and per dollar) needs a success count, so like the
rate it isn't given when verdicts are undecided. Tokens are agent plus customer when every
conversation has the customer's count, and agent only otherwise (``tokens_counted`` says which):
never a mix, so conversations count the same way.

**Per dollar** counts API spend only (the agent's and the customer's). A self-hosted student
(the pod arms) costs $0 there, so its successes per dollar can't be compared with an API arm's;
successes per million tokens can.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable
from functools import cache
from typing import Any

from reflexive.files import latest_records
from reflexive.hallucination import agent_view_of, domain_tools
from reflexive.metrics import is_broken, record_cost
from reflexive.stats import cluster_bootstrap

Record = dict[str, Any]

SCORERS = ("tool_call_validity", "redundant_tool_calls", "agent_loop_detection")
JUDGED = "tool_selection"
REPS = 10_000
UNDECIDED = ("unknown", "error")  # verdicts that could be a success or a failure
COST_COVERS = "API calls only (agent and customer): a self-hosted student counts as $0"


@cache
def tool_schemas(domain: str) -> tuple[dict, ...]:
    """The domain's agent tools as OpenAI function schemas (what the agent was offered).

    From ``configs/tool_schemas.json`` (saved from τ²-bench; ``just`` checks it with
    ``scripts/tau2/dump_tool_schemas.py --check``), so scoring needs no τ²-bench."""
    return tuple(domain_tools(domain)[0])


def agent_scores(rec: Record, schemas: Iterable[dict], judge: Any = None) -> dict[str, Any]:
    """AuditKIT's agent scorers on one conversation: {scorer: value or None, ...}.

    None when a scorer has nothing to measure (no tool calls). ``judge``: an AuditKIT
    ``ToolSelectionJudge`` for ``tool_selection``, or None to skip it."""
    from auditkit.metrics.agent import AgentLoopDetection, RedundantToolCalls, ToolCallValidity
    from auditkit.sample import Sample

    msgs = agent_view_of(rec) if rec.get("agent_view") or rec.get("messages") else []
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
    cost = record_cost(rec)
    return {
        "agent_tokens": agent_tokens,
        "agent_calls": au.get("calls"),
        "user_tokens": user_tokens,
        "total_tokens": agent_tokens + (user_tokens or 0) if user_tokens is not None else None,
        "cost_usd": round(cost, 5) if cost is not None else None,
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
        "arm": rec.get("arm"),
        "verdict": success(rec),
        "termination": rec.get("termination"),
        **agent_scores(rec, tool_schemas(rec["domain"]) if rec.get("domain") else (), judge),
        **ttc(rec),
    }


def saved_verdict(rec: Record) -> str:
    """The run's own verdict. A pass with judged checks skipped (judges off) isn't verified: unknown,
    as `reflexive.oracle` has it."""
    o = rec.get("outcome") or {}
    if not o.get("passed"):
        return "failure"
    return "unknown" if o.get("skipped") else "success"


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
    """95% range of the success rate, resampling tasks (all their samples): `cluster_bootstrap`."""
    by_task: dict[str, list[bool]] = {}
    for i, r in enumerate(rows):
        if r["verdict"] in ("success", "failure"):
            by_task.setdefault(r.get("task_id") or f"row{i}", []).append(r["verdict"] == "success")
    got = cluster_bootstrap(by_task, reps, seed)
    return [round(got[0], 4), round(got[1], 4)] if got else None


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


def summarise(rows: list[dict[str, Any]], crashed: int = 0, crashed_cost: float = 0.0) -> dict[str, Any]:
    """An arm's headline numbers from its conversations' `evaluate_record` rows.

    ``crashed_cost``: the spend of the crashed conversations, which counts in the cost."""
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
    with_user = bool(rows) and all(r.get("user_tokens") is not None for r in rows)
    tokens = sum(r["agent_tokens"] + (r["user_tokens"] if with_user else 0) for r in rows)
    cost = sum(r["cost_usd"] or 0 for r in rows) + crashed_cost
    out["cost_usd_total"] = round(cost, 4)
    out["crashed_cost_usd"] = round(crashed_cost, 4)
    out["tokens_counted"] = "agent and customer" if with_user else "agent only"
    # a success count is needed: with undecided verdicts there's none, only bounds
    out["successes_per_million_tokens"] = round(n_ok / tokens * 1e6, 3) if rated and tokens else None
    out["successes_per_dollar"] = round(n_ok / cost, 3) if rated and cost else None
    out["cost_covers"] = COST_COVERS
    out["user_tokens_recorded"] = with_user if rows else None
    return out


def evaluate(
    records: list[Record], success: Callable[[Record], str], judge: Any = None, workers: int = 1
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """(one row per scored conversation, the summary) for a run's records.

    Rows run ``workers`` at a time, as ``oracle-score`` does: each oracle verdict builds its own
    τ²-bench environment. The shared caches (tool schemas; the oracle's per-domain scorer, see
    `warm`) are filled first, in this thread, so threads never build them at once."""
    from concurrent.futures import ThreadPoolExecutor

    recs = latest_records(records)
    broken = [r for r in recs if is_broken(r)]
    ok = [r for r in recs if not is_broken(r)]
    warm({r["domain"] for r in ok if r.get("domain")})
    with ThreadPoolExecutor(max(1, workers)) as pool:
        rows = list(pool.map(lambda r: evaluate_record(r, success, judge), ok))
    crashed_cost = sum(record_cost(r) or 0 for r in broken)
    return rows, summarise(rows, crashed=len(broken), crashed_cost=crashed_cost)


def warm(domains: Iterable[str], oracle: bool = False) -> None:
    """Fill the per-domain caches before threads use them (`functools.cache` doesn't lock).

    ``oracle``: also load τ²-bench's scorer for each domain (`reflexive.outcome.scorer`)."""
    for d in domains:
        tool_schemas(d)
        if oracle:
            from reflexive.outcome import scorer

            scorer(d)
