"""The numbers that summarise one arm's run on one domain and split (SEL-55, then SEL-38).

``arm_metrics(records)`` gives the fields of Chiara's ``results/arm-A-telecom-train.metrics.json``
(4 Nov), computed from the rollout records instead of by hand. Pure: no API calls, no τ²-bench.

Definitions:
- A **crashed** record (``error`` set: a rate limit, a server error) isn't scored, and counts in
  neither ``passed`` nor the success rate. A re-run of the arm retries it.
- **task_success** is passed ÷ scored conversations, with a 95% Wilson range.
- **conversations_flagged** uses the heuristic diagnostics every record carries
  (``reflexive.teacher``: unknown tools, ID-like arguments never seen before, a failed call
  repeated unchanged), not the Tool / Arg / Loop classifier (``reflexive.hallucination``).
- **transferred_to_human**: conversations where the agent called ``transfer_to_human_agents``.
- **What counts crashed records:** only the money and the calls (``student_calls``,
  ``student_replies_cut_off``, ``cost_usd``), which were spent either way. Every rate,
  count and per-conversation figure uses the scored conversations only.
"""

from __future__ import annotations

import collections
import re
import statistics
from collections.abc import Iterable
from typing import Any

from reflexive.stats import wilson

Record = dict[str, Any]

FLAG_METHOD = (
    "heuristic diagnostics (reflexive.teacher.unseen_arguments / tool_errors), not the SEL-12 classifier"
)
_ISSUE = re.compile(r"^\[(\w+?)\]")  # telecom task IDs start with the issue: "[mms_issue]…"


def _transferred(rec: Record) -> bool:
    return any(
        tc.get("function", {}).get("name") == "transfer_to_human_agents"
        for m in rec.get("messages") or []
        if m.get("role") == "assistant"
        for tc in m.get("tool_calls") or []
    )


def _flags(rec: Record) -> dict[str, bool]:
    d = rec.get("diagnostics") or {}
    tool, arg = bool(d.get("unknown_tools")), bool(d.get("unseen_arguments"))
    return {
        "tool_unknown_tool": tool,
        "arg_unseen_id_like_argument": arg,
        "loop_repeated_failed_call": bool(d.get("repeated_failures")),
        "tool_or_arg": tool or arg,
    }


def by_issue(recs: Iterable[Record]) -> dict[str, dict[str, int]] | None:
    """Passes per telecom issue type (mms, mobile data, service); None for other domains."""
    out: dict[str, dict[str, int]] = {}
    for r in recs:
        m = _ISSUE.match(str(r.get("task_id")))
        if r.get("domain") != "telecom" or not m:
            continue
        row = out.setdefault(m.group(1), {"passed": 0, "n": 0})
        row["passed"] += bool(r["outcome"]["passed"])
        row["n"] += 1
    return dict(sorted(out.items())) or None


def arm_metrics(records: list[Record]) -> dict[str, Any]:
    """Success, endings, hallucination flags, calls, tokens and cost for one run's records."""
    scored = [r for r in records if not r.get("error")]
    n, k = len(scored), sum(bool(r["outcome"]["passed"]) for r in scored)
    usage = [r.get("agent_usage") or {} for r in records]
    calls = sum(u.get("calls", 0) for u in usage)
    flagged = collections.Counter()
    for r in scored:
        flagged.update(name for name, on in _flags(r).items() if on)
    customer = sum(r.get("user_cost") or 0 for r in records)
    student = sum(u.get("est_cost", 0) for u in usage)
    lo, hi = wilson(k, n)
    return {
        "conversations": n,
        "crashed": len(records) - n,
        "passed": k,
        "task_success": round(k / n, 4) if n else None,
        "task_success_wilson95": [round(lo, 4), round(hi, 4)],
        "by_issue": by_issue(scored),
        "termination": dict(collections.Counter(r.get("termination") for r in scored).most_common()),
        "context_overflow": sum(bool((r.get("flags") or {}).get("context_overflow")) for r in scored),
        "conversations_flagged": {
            name: flagged[name]
            for name in (
                "tool_unknown_tool",
                "arg_unseen_id_like_argument",
                "loop_repeated_failed_call",
                "tool_or_arg",
            )
        },
        "flag_method": FLAG_METHOD,
        "hallucination_rate_tool_or_arg": round(flagged["tool_or_arg"] / n, 4) if n else None,
        "transferred_to_human": sum(_transferred(r) for r in scored),
        "judged": sum(bool(r["outcome"].get("judged")) for r in scored),
        "skipped_checks": sum(bool(r["outcome"].get("skipped")) for r in scored),
        "student_calls": calls,
        "student_calls_per_success": round(calls / k, 1) if k else None,
        "student_replies_cut_off": sum(u.get("cut_off", 0) for u in usage),
        "tool_calls_per_conversation": (
            round(statistics.mean((r.get("diagnostics") or {}).get("tools_ran", 0) for r in scored), 1)
            if n
            else None
        ),
        "student_tokens_per_conversation_median": (
            int(
                statistics.median(
                    (r.get("agent_usage") or {}).get("prompt_tokens", 0)
                    + (r.get("agent_usage") or {}).get("completion_tokens", 0)
                    for r in scored
                )
            )
            if n
            else None
        ),
        "cost_usd": {
            "customer": round(customer, 2),
            "student_est": round(student, 2),
            "total": round(customer + student, 2),
        },
    }
