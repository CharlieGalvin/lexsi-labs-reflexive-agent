"""On-policy teacher fixes: where the student went wrong on train tasks, the teacher's turn instead.

    python scripts/heal/onpolicy_fixes.py results/episodes/arms/S2-*-trainpool.jsonl --dry-run
    python scripts/heal/onpolicy_fixes.py results/episodes/arms/S2-*-trainpool.jsonl

The heal loop's `PrefixTeacher` fixes a failure at the classifier's first finding (a Tool, Arg or
Loop hallucination). Most of SFT v2's remaining failures have none: they are wrong *decisions*
(an unneeded change, wrong arguments, a change never made). On a train task the expected actions
are known, so this script finds the turn to fix from them:

1. **The first unexpected change:** the first agent turn with a state-changing call that matches
   no expected action still pending (τ²-bench's own matching rule). An unneeded change, or the
   right tool with the wrong arguments.
2. **A change never made:** no unexpected change, but an expected action is still pending at the
   end. The fix point is the agent's last turn, where it stopped short.
3. **Otherwise the classifier's first finding** (a loop or a hallucination), as `PrefixTeacher` does.

The teacher (GLM-5.3-Flash, as everywhere) writes that turn again from the conversation up to it,
up to three times (temperatures 0.0, 0.7, 0.7). A turn is accepted only if `heal.check_fix`
passes it **and** every state-changing call in it matches an expected action still pending. A
turn with no change at all (asking the customer to confirm, explaining a policy) passes that
check, except at a change-never-made point, where it must make one of the pending changes.

Only train tasks are fixed, and never the 30 validation tasks (``--exclude``, by default the task
IDs in ``results/episodes/arms/P-*-val.jsonl``). Each accepted turn becomes a ``reagent.sft/v1``
example (group ``onpolicy``, the fixed turn the only target), like the heal loop's. Writes
``--out`` and a report beside it (``.report.json``: what each failure was, every attempt and why
it was refused). ``--dry-run`` only finds the fix points: no API call.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, latest_records, read_jsonl, shown


def _failure_diff() -> Any:
    """scripts/arms/failure_diff.py, for τ²-bench's matching rule (`same_call`): one copy of it."""
    path = Path(__file__).resolve().parents[1] / "arms" / "failure_diff.py"
    spec = importlib.util.spec_from_file_location("failure_diff", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("failure_diff", mod)
    spec.loader.exec_module(mod)
    return mod


same_call = _failure_diff().same_call

Call = tuple[str, dict]
Expected = tuple[str, dict, Any]  # (name, arguments, compare_args)
VAL = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "heal" / "onpolicy-fixes.jsonl"


def calls_in(msg: dict) -> list[Call]:
    """The tool calls of one assistant message, arguments parsed."""
    out = []
    for tc in msg.get("tool_calls") or []:
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            args = {"__raw__": raw}
        out.append((tc["function"]["name"], args if isinstance(args, dict) else {"__raw__": args}))
    return out


@dataclass(frozen=True)
class FixPoint:
    turn: int  # index in the agent's view of the assistant turn to rewrite
    kind: str  # "unexpected_change" | "change_never_made" | "classifier"
    detail: str
    pending: tuple[Expected, ...]  # expected changes not yet made before that turn


def locate(view: list[dict], expected: list[Expected], mutating: Callable[[str], bool]) -> FixPoint | None:
    """The first turn where the agent's changes leave the expected ones (rules 1 and 2), if any."""
    pending = list(expected)
    last_agent = None
    for i, msg in enumerate(view):
        if msg.get("role") != "assistant":
            continue
        last_agent = i
        before = tuple(pending)
        for call in calls_in(msg):
            if not mutating(call[0]):
                continue
            hit = next((e for e in pending if same_call(e, call)), None)
            if hit is None:
                same = [e for e in pending if e[0] == call[0]]
                what = "wrong arguments" if same else "not expected"
                return FixPoint(i, "unexpected_change", f"{call[0]}: {what}", before)
            pending.remove(hit)
    if pending and last_agent is not None and last_agent > 0:
        names = ", ".join(e[0] for e in pending)
        return FixPoint(last_agent, "change_never_made", f"never done: {names}", tuple(pending))
    return None


def pending_at(
    view: list[dict], turn: int, expected: list[Expected], mutating: Callable[[str], bool]
) -> tuple[Expected, ...]:
    """The expected changes the agent hasn't made before ``turn``."""
    pending = list(expected)
    for msg in view[:turn]:
        if msg.get("role") == "assistant":
            for call in calls_in(msg):
                hit = next((e for e in pending if mutating(call[0]) and same_call(e, call)), None)
                if hit is not None:
                    pending.remove(hit)
    return tuple(pending)


def verify(fixed: dict, point: FixPoint, mutating: Callable[[str], bool]) -> list[str]:
    """Why the teacher's turn doesn't fix the decision (empty if it does)."""
    problems = []
    writes = [c for c in calls_in(fixed) if mutating(c[0])]
    left = list(point.pending)
    for call in writes:
        hit = next((e for e in left if same_call(e, call)), None)
        if hit is None:
            problems.append(f"{call[0]} matches no expected change still pending")
        else:
            left.remove(hit)
    if point.kind == "change_never_made" and not writes:
        problems.append("at a change-never-made point the fix must make one of the pending changes")
    return problems


# --- With τ²-bench: tasks, tools and the teacher -----------------------------------------------


def teach(
    failure: Any, point: FixPoint, mutating: Callable[[str], bool], attempts: int
) -> tuple[dict | None, list]:
    """The teacher's turn at ``point``, up to ``attempts`` tries: the accepted example, and every try."""
    from reflexive import heal

    domain = failure.task[0]
    try:
        prefix, failed = heal.fix_point(failure)
    except ValueError as e:
        return None, [{"accepted": False, "problems": [str(e)]}]
    schemas, _ = heal.domain_tools(domain)
    log: list[dict] = []
    for i in range(1, attempts + 1):
        temp = heal.TEMPERATURES[min(i, len(heal.TEMPERATURES)) - 1]
        agent = heal.PrefixTeacher()._agent(temp)
        start = time.monotonic()
        try:
            fixed = heal._clean_turn(agent.turn([dict(m) for m in prefix], schemas))
        except Exception as e:  # noqa: BLE001 - an API failure is a refused attempt
            log.append({"attempt": i, "accepted": False, "error": f"{type(e).__name__}: {e}"[:300]})
            continue
        problems = heal.check_fix(domain, prefix, fixed, failed) + verify(fixed, point, mutating)
        log.append(
            {
                "attempt": i,
                "temperature": temp,
                "accepted": not problems,
                "problems": problems,
                "turn": fixed,
                "seconds": round(time.monotonic() - start, 2),
                "usage": dict(getattr(agent, "usage", {}) or {}),
            }
        )
        if not problems:
            meta = {
                "teacher": heal.TEACHER,
                "attempt": i,
                "temperature": temp,
                "failed_turn": heal._clean_turn(failed),
                "fixed_turn_message": fixed,
            }
            ex = heal.to_sft_example(failure, prefix, fixed, meta)
            ex["id"] = ex["id"].replace("heal/", "onpolicy/", 1)
            ex["group"] = "onpolicy"
            ex["source"].update(kind="onpolicy_fix", fix_kind=point.kind, fix_detail=point.detail)
            return ex, log
    return None, log


def val_ids(paths: Iterable[str]) -> set[tuple[str, str]]:
    return {
        (r["domain"], str(r["task_id"])) for p in paths for r in read_jsonl(Path(p), partial_last_ok=True)
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path, help="rollout files of the student on train tasks")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--exclude",
        nargs="*",
        default=sorted(glob.glob(str(VAL / "P-*-val.jsonl"))),
        help="rollout files whose tasks are never fixed (default: the validation tasks)",
    )
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="find the fix points only: no API call")
    a = p.parse_args()

    from loguru import logger
    from tau2.registry import registry

    from reflexive.agent import Failure
    from reflexive.hallucination import Diagnosis, FailureType, Finding, agent_view_of, classify_record
    from reflexive.metrics import is_broken
    from reflexive.outcome import scorer
    from reflexive.splits import split_of

    logger.remove()
    if not a.exclude:
        print("⛔ no validation files to exclude (--exclude): refusing to risk fixing validation tasks")
        return 1
    banned = val_ids(a.exclude)
    envs, tasks = {}, {}
    report: dict[str, Any] = {
        "files": [shown(f) for f in a.files],
        "excluded": len(banned),
        "conversations": [],
    }
    kinds: collections.Counter = collections.Counter()
    fixes = []

    for f in a.files:
        for r in latest_records(read_jsonl(f, partial_last_ok=True)):
            d, t = r["domain"], str(r["task_id"])
            if (r.get("outcome") or {}).get("passed") or is_broken(r):
                continue
            if (d, t) in banned or split_of(d, t) != "train":
                kinds["left out: a validation or non-train task"] += 1
                continue
            if d not in envs:
                tasks[d] = scorer(d)
                envs[d] = registry.get_env_constructor(d)(**(tasks[d].env_kwargs or {})).tools
            tools = envs[d]

            def mutating(name: str, tools: Any = tools) -> bool:
                return tools.has_tool(name) and bool(tools.tool_mutates_state(name))

            ec = tasks[d].task(t).evaluation_criteria
            expected = [
                (x.name, x.arguments or {}, x.compare_args) for x in ec.actions or [] if x.requestor != "user"
            ]
            view = agent_view_of(r)
            writes = [e for e in expected if mutating(e[0])]
            point = locate(view, writes, mutating)
            diag = classify_record(r)
            if point is None and diag.first is not None:
                point = FixPoint(
                    diag.first.turn,
                    "classifier",
                    f"{diag.first.type.value}/{diag.first.kind}",
                    pending_at(view, diag.first.turn, writes, mutating),
                )
            entry: dict[str, Any] = {"domain": d, "task_id": t, "sample": r.get("sample", 0)}
            if point is None:
                kinds["no fix point (actions match; failed another check)"] += 1
                report["conversations"].append({**entry, "fix_point": None})
                continue
            kinds[point.kind] += 1
            entry["fix_point"] = {"turn": point.turn, "kind": point.kind, "detail": point.detail}
            failure = Failure(
                r,
                Diagnosis(
                    diag.failure if point.kind == "classifier" else FailureType.NONE,
                    [
                        Finding(
                            diag.first.type
                            if point.kind == "classifier" and diag.first
                            else FailureType.NONE,
                            point.kind,
                            point.turn,
                            detail=point.detail,
                        )
                    ],
                ),
            )
            if a.dry_run:
                report["conversations"].append(entry)
                continue
            ex, entry["attempts"] = teach(failure, point, mutating, a.attempts)
            if ex is not None:
                fixes.append(ex)
            report["conversations"].append(entry)

    report["fix_points"] = dict(kinds)
    report["fixes"] = len(fixes)
    report["fixes_by_kind"] = dict(collections.Counter(x["source"]["fix_kind"] for x in fixes))
    print(f"fix points: {dict(kinds)}")
    if a.dry_run:
        print("(dry run: no teacher calls, nothing written)")
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in fixes))
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"accepted fixes: {len(fixes)} {report['fixes_by_kind']} → {shown(a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
