"""Bootstrap confidence intervals for comparing τ²-bench runs (SEL-27; SEL-11 applies them).

    a, b = read_jsonl(Path("…/A-telecom-train.jsonl")), read_jsonl(Path("…/C-telecom-train.jsonl"))
    table = compare(a, b)                 # each metric: both arms' values, 95% ranges, difference
    print(to_markdown(table, "A", "C"))

**The five metrics**, one value per scored conversation (a crashed one, `metrics.is_broken`,
says nothing about the model and is left out, as in `metrics.arm_metrics`):

| Metric | Per conversation | Run value |
|---|---|---|
| ``success_rate`` | 1 if τ²-bench passed it | share passed |
| ``hallucination_rate`` | 1 if the classifier finds a Tool or Arg failure | share with one |
| ``loop_rate`` | 1 if the classifier finds a Loop failure | share with one |
| ``steps_per_task`` | the agent's turns | mean |
| ``redundant_call_rate`` | exact repeats of an earlier call ÷ tool calls | all repeats ÷ all calls |

Hallucination and loops use the Tool / Arg / Loop classifier (`hallucination.classify_record`,
SEL-12), the one the gate uses, not the heuristic flags in ``metrics.arm_metrics``.

**How the ranges are made.** Every metric is a ratio of sums over conversations (numerator ÷
denominator; the denominator is 1 per conversation except for redundant calls), so one method
covers them all:

- **one run:** resample its conversations with replacement, ``reps`` times, and take the 2.5th
  and 97.5th percentiles of the metric (percentile bootstrap);
- **the difference between two runs** (b − a): **paired** on the tasks both ran. Resample those
  tasks, and take each task's sums in both runs, so a hard task counts in both arms at once. With
  several samples per task, a task's conversations stay together (a cluster bootstrap).

An exact repeat is the same tool with the same arguments, keys in any order.

A range that excludes 0 is a difference this sample can see; with tens of tasks the ranges are
wide, which is the point of showing them. Stdlib only; a fixed seed gives the same ranges.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from reflexive.hallucination import FailureType, agent_view_of, classify_record
from reflexive.metrics import is_broken, monitor_stopped

Record = dict[str, Any]
Task = tuple[str, str]
Pair = tuple[float, float]  # (numerator, denominator) of one conversation

REPS = 10_000
METRICS: dict[str, str] = {
    "success_rate": "Success rate",
    "hallucination_rate": "Hallucination rate (Tool or Arg)",
    "loop_rate": "Loop rate",
    "steps_per_task": "Steps per task (agent turns)",
    "redundant_call_rate": "Redundant-call rate",
}
HIGHER_IS_BETTER = {"success_rate": True}  # the others: lower is better


# --- Per conversation --------------------------------------------------------------------------


def _calls(rec: Record) -> list[tuple[str, str]]:
    """The agent's tool calls, in order, as (tool, arguments with keys sorted)."""
    out = []
    for m in agent_view_of(rec):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.dumps(json.loads(raw) if isinstance(raw, str) else raw, sort_keys=True)
            except (json.JSONDecodeError, TypeError):
                args = str(raw)
            out.append((str(fn.get("name")), args))
    return out


def redundant_calls(rec: Record) -> tuple[int, int]:
    """(calls that repeat an earlier call exactly, all tool calls) in one conversation."""
    calls = _calls(rec)
    return len(calls) - len(set(calls)), len(calls)


def steps(rec: Record) -> int:
    """The agent's turns: each reply, whether a message or tool calls."""
    return sum(m.get("role") == "assistant" for m in agent_view_of(rec))


def conversation_values(rec: Record) -> dict[str, Pair]:
    """Each metric's (numerator, denominator) for one scored conversation."""
    types = classify_record(rec).types
    red, calls = redundant_calls(rec)
    return {
        "success_rate": (float(bool((rec.get("outcome") or {}).get("passed"))), 1.0),
        "hallucination_rate": (float(bool(types & {FailureType.TOOL, FailureType.ARG})), 1.0),
        "loop_rate": (float(FailureType.LOOP in types), 1.0),
        "steps_per_task": (float(steps(rec)), 1.0),
        "redundant_call_rate": (float(red), float(calls)),
    }


# --- The bootstrap -----------------------------------------------------------------------------


def _ratio(pairs: Sequence[Pair]) -> float | None:
    den = sum(d for _, d in pairs)
    return sum(n for n, _ in pairs) / den if den else None


def _percentiles(values: list[float]) -> tuple[float, float] | None:
    if not values:
        return None
    values.sort()
    k = len(values)
    return values[int(0.025 * k)], values[min(int(0.975 * k), k - 1)]


def bootstrap(pairs: Sequence[Pair], reps: int = REPS, seed: int = 0) -> tuple[float, float] | None:
    """95% percentile-bootstrap range of sum(num) ÷ sum(den), resampling the conversations."""
    if not pairs:
        return None
    rng = random.Random(seed)
    stats = [r for _ in range(reps) if (r := _ratio(rng.choices(pairs, k=len(pairs)))) is not None]
    return _percentiles(stats)


def paired_bootstrap_diff(
    a: dict[Task, list[Pair]], b: dict[Task, list[Pair]], reps: int = REPS, seed: int = 0
) -> tuple[float, float] | None:
    """95% range of ratio(b) − ratio(a), resampling the tasks both ran (their conversations together)."""
    shared = sorted(a.keys() & b.keys())
    if not shared:
        return None
    sums_a = [_sums(a[t]) for t in shared]
    sums_b = [_sums(b[t]) for t in shared]
    rng = random.Random(seed)
    idx = range(len(shared))
    diffs = []
    for _ in range(reps):
        pick = rng.choices(idx, k=len(shared))
        ra, rb = _ratio([sums_a[i] for i in pick]), _ratio([sums_b[i] for i in pick])
        if ra is not None and rb is not None:
            diffs.append(rb - ra)
    return _percentiles(diffs)


def _sums(pairs: list[Pair]) -> Pair:
    return sum(n for n, _ in pairs), sum(d for _, d in pairs)


# --- Comparing two runs ------------------------------------------------------------------------


@dataclass
class Row:
    metric: str
    a: float | None
    a_ci: tuple[float, float] | None
    b: float | None
    b_ci: tuple[float, float] | None
    diff: float | None  # b − a, on the shared tasks
    diff_ci: tuple[float, float] | None

    @property
    def clear(self) -> bool:
        """The difference's range excludes 0: a difference this sample can see."""
        return self.diff_ci is not None and (self.diff_ci[0] > 0 or self.diff_ci[1] < 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "a": self.a,
            "a_ci95": self.a_ci,
            "b": self.b,
            "b_ci95": self.b_ci,
            "diff_b_minus_a": self.diff,
            "diff_ci95": self.diff_ci,
            "clear": self.clear,
        }


@dataclass
class Comparison:
    rows: list[Row]
    a_scored: int
    b_scored: int
    a_crashed: int
    b_crashed: int
    shared_tasks: int
    reps: int
    seed: int
    a_monitor_stopped: dict[str, Any] | None = None  # metrics.monitor_stopped: the guard's stops
    b_monitor_stopped: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "a_scored": self.a_scored,
            "b_scored": self.b_scored,
            "a_crashed": self.a_crashed,
            "b_crashed": self.b_crashed,
            "shared_tasks": self.shared_tasks,
            "reps": self.reps,
            "seed": self.seed,
            "a_monitor_stopped": self.a_monitor_stopped,
            "b_monitor_stopped": self.b_monitor_stopped,
            "method": "percentile bootstrap; difference paired on shared tasks (cluster by task)",
        }


def _by_task(
    recs: Sequence[Record], values: Callable[[Record], dict[str, Pair]]
) -> dict[str, dict[Task, list[Pair]]]:
    out: dict[str, dict[Task, list[Pair]]] = {m: {} for m in METRICS}
    for r in recs:
        task = (str(r["domain"]), str(r["task_id"]))
        for m, p in values(r).items():
            out[m].setdefault(task, []).append(p)
    return out


def compare(
    a: Sequence[Record],
    b: Sequence[Record],
    *,
    reps: int = REPS,
    seed: int = 0,
    values: Callable[[Record], dict[str, Pair]] = conversation_values,
) -> Comparison:
    """Each metric for runs ``a`` and ``b``, with 95% ranges, and the paired difference b − a."""
    sa, sb = [r for r in a if not is_broken(r)], [r for r in b if not is_broken(r)]
    if not sa or not sb:
        raise ValueError("a run with no scored conversations: nothing to compare")
    ta, tb = _by_task(sa, values), _by_task(sb, values)
    shared = ta["success_rate"].keys() & tb["success_rate"].keys()
    rows = []
    for i, m in enumerate(METRICS):
        pa = [p for ps in ta[m].values() for p in ps]
        pb = [p for ps in tb[m].values() for p in ps]
        on_a = [p for t in shared for p in ta[m][t]]
        on_b = [p for t in shared for p in tb[m][t]]
        ra, rb = _ratio(on_a), _ratio(on_b)
        rows.append(
            Row(
                metric=m,
                a=_ratio(pa),
                a_ci=bootstrap(pa, reps, seed + i),
                b=_ratio(pb),
                b_ci=bootstrap(pb, reps, seed + i),
                diff=None if ra is None or rb is None else rb - ra,
                diff_ci=paired_bootstrap_diff(ta[m], tb[m], reps, seed + i),
            )
        )
    return Comparison(
        rows,
        len(sa),
        len(sb),
        len(a) - len(sa),
        len(b) - len(sb),
        len(shared),
        reps,
        seed,
        monitor_stopped(sa),
        monitor_stopped(sb),
    )


# --- Showing it --------------------------------------------------------------------------------


def _fmt(metric: str, x: float | None) -> str:
    if x is None:
        return "–"
    return f"{x:.1f}" if metric == "steps_per_task" else f"{x:.1%}"


def _fmt_diff(metric: str, x: float | None) -> str:
    if x is None:
        return "–"
    return f"{x:+.1f}" if metric == "steps_per_task" else f"{x * 100:+.1f} pts"


def _fmt_ci(metric: str, ci: tuple[float, float] | None, diff: bool = False) -> str:
    if ci is None:
        return "–"
    f = _fmt_diff if diff else _fmt
    return f"{f(metric, ci[0])} to {f(metric, ci[1])}"


def to_markdown(c: Comparison, name_a: str, name_b: str) -> str:
    """The comparison as a table: each run with its 95% range, and the paired difference."""
    lines = [
        f"| Metric | {name_a} | 95% CI | {name_b} | 95% CI | {name_b} − {name_a} (paired) | 95% CI | |",
        "|---|---:|---|---:|---|---:|---|---|",
    ]
    for r in c.rows:
        mark = ""
        if r.clear and r.diff is not None:
            better = (r.diff > 0) == HIGHER_IS_BETTER.get(r.metric, False)
            mark = f"{name_b} {'better' if better else 'worse'}"
        lines.append(
            f"| {METRICS[r.metric]} | {_fmt(r.metric, r.a)} | {_fmt_ci(r.metric, r.a_ci)} "
            f"| {_fmt(r.metric, r.b)} | {_fmt_ci(r.metric, r.b_ci)} "
            f"| {_fmt_diff(r.metric, r.diff)} | {_fmt_ci(r.metric, r.diff_ci, diff=True)} | {mark} |"
        )
    lines += [
        "",
        f"{c.a_scored} vs {c.b_scored} scored conversations ({c.a_crashed} and {c.b_crashed} crashed, "
        f"left out); the difference is paired on the {c.shared_tasks} tasks both ran. 95% percentile "
        f"bootstrap, {c.reps:,} resamples, seed {c.seed}. A mark means the difference's range excludes 0.",
    ]
    for name, ms in ((name_a, c.a_monitor_stopped), (name_b, c.b_monitor_stopped)):
        if ms and ms["conversations"]:
            lines.append(
                f"\n{name}: the runtime monitor's guard ended {ms['conversations']} conversations "
                f"(termination monitor_stop), {ms['passed']} of them passed; counting those as "
                f"failures, success is {ms['task_success_if_failed']:.1%}."
            )
    return "\n".join(lines)
