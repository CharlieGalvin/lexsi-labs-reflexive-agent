"""Headline metrics for the before/after comparison.

These are computed with the same parser the rewards use. AuditKIT's scorers are run
alongside (see scripts/evaluate.py) as an independent check, so we are not only
grading ourselves with our own reward code.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from . import trajectory as T


def episode_metrics(messages: list[dict[str, Any]], gold: str | None) -> dict[str, Any]:
    traj = T.parse(messages)
    unanswerable = gold is not None and str(gold).upper() == T.UNANSWERABLE
    answered = bool(traj.final_answer) and not traj.abstained
    hallucinated = (answered and (unanswerable or not traj.successful_results
                                  or bool(T.unsupported_numbers(traj.final_answer,
                                                                traj.successful_results))))
    return {
        "correct": T.is_correct(traj.final_answer, gold),
        "tool_calls": traj.n_calls,
        "redundant_calls": traj.n_redundant,
        "failed_calls": traj.n_errors,
        "malformed_calls": traj.n_malformed,
        "answered": answered,
        "abstained": traj.abstained,
        "no_answer": not traj.final_answer,
        "hallucinated": hallucinated,
        "unanswerable": unanswerable,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    ans = [r for r in rows if not r["unanswerable"]]
    unans = [r for r in rows if r["unanswerable"]]
    calls = sum(r["tool_calls"] for r in rows)

    def rate(xs, key):
        return round(mean(float(x[key]) for x in xs), 4) if xs else float("nan")

    return {
        "n": len(rows),
        "task_success": rate(rows, "correct"),
        "accuracy_answerable": rate(ans, "correct"),
        "correct_abstention_unanswerable": rate(unans, "abstained"),
        "hallucination_rate": rate(rows, "hallucinated"),
        "avg_tool_calls": round(calls / len(rows), 3),
        "redundant_call_rate": round(sum(r["redundant_calls"] for r in rows) / calls, 4) if calls else 0.0,
        "failed_call_rate": round(sum(r["failed_calls"] for r in rows) / calls, 4) if calls else 0.0,
        "malformed_call_rate": round(sum(r["malformed_calls"] for r in rows) / calls, 4) if calls else 0.0,
        "no_answer_rate": rate(rows, "no_answer"),
        # calls per *successful* task: the efficiency number to beat without losing success
        "calls_per_success": round(calls / max(1, sum(r["correct"] for r in rows)), 3),
    }
