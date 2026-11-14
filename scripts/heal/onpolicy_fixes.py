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

A state-changing call whose tool result is an error changed nothing: it neither counts as an
unexpected change nor makes an expected one.

Only train tasks are fixed, and never the 30 validation tasks (``--exclude``, by default the task
IDs in ``results/episodes/arms/P-*-val.jsonl``). The script refuses to run unless those files
are whole and hold all 30 (13 retail, 5 airline, 12 telecom): a missing or cut-short file would
let validation tasks into the training data. Each accepted turn becomes a ``reagent.sft/v1``
example (group ``onpolicy``, the fixed turn the only target), like the heal loop's. Writes
``--out`` and a report beside it (``.report.json``: what each failure was, every attempt and why
it was refused). Each accepted example is written as soon as it is accepted and the report is
written however the run ends, so a crash or Ctrl-C keeps what was paid for. Running out of
OpenRouter credit stops the run (the rest would only fail). ``--dry-run`` only finds the fix
points: no API call.
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

from reflexive.files import ROOT, BadJsonLine, latest_records, read_jsonl, shown
from reflexive.hallucination import Diagnosis, FailureType, Finding


def _failure_diff() -> Any:
    """scripts/arms/failure_diff.py, for τ²-bench's matching rule (`same_call`): one copy of it."""
    path = Path(__file__).resolve().parents[1] / "arms" / "failure_diff.py"
    spec = importlib.util.spec_from_file_location("failure_diff", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("failure_diff", mod)
    spec.loader.exec_module(mod)
    return mod


_fd = _failure_diff()
same_call, agent_calls = _fd.same_call, _fd.agent_calls

Call = tuple[str, dict]
Expected = tuple[str, dict, Any]  # (name, arguments, compare_args)
VAL = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "heal" / "onpolicy-fixes.jsonl"
VAL_TASKS = {"retail": 13, "airline": 5, "telecom": 12}  # the 30 validation tasks (SEL-71)


def calls_in(msg: dict) -> list[Call]:
    """The tool calls of one assistant message, arguments parsed (`failure_diff.agent_calls`)."""
    return [(n, a if isinstance(a, dict) else {"<unparsed>": a}) for n, a in agent_calls([msg])]


def changes_in(view: list[dict], i: int, mutating: Callable[[str], bool]) -> list[Call]:
    """The state-changing calls of the assistant turn ``view[i]`` that the environment carried out.

    A call whose result (the tool messages right after the turn) starts with "Error" changed
    nothing, so it isn't a change."""
    results = {}
    for m in view[i + 1 :]:
        if m.get("role") != "tool":
            break
        results[m.get("tool_call_id")] = str(m.get("content") or "")
    ids = [tc.get("id") for tc in view[i].get("tool_calls") or []]
    return [
        call
        for call, cid in zip(calls_in(view[i]), ids, strict=False)
        if mutating(call[0]) and not results.get(cid, "").startswith("Error")
    ]


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
        for call in changes_in(view, i, mutating):
            hit = next((e for e in pending if same_call(e, call)), None)
            if hit is None:
                same = [e for e in pending if e[0] == call[0]]
                what = "wrong arguments" if same else "not expected"
                return FixPoint(i, "unexpected_change", f"{call[0]}: {what}", before)
            pending.remove(hit)
    if pending and last_agent is not None and last_agent > 0:
        names = ", ".join(e[0] for e in pending)
        # pending *before* the last turn: that turn is the one rewritten, so its changes are redone
        before_last = pending_at(view, last_agent, expected, mutating)
        return FixPoint(last_agent, "change_never_made", f"never done: {names}", before_last)
    return None


def pending_at(
    view: list[dict], turn: int, expected: list[Expected], mutating: Callable[[str], bool]
) -> tuple[Expected, ...]:
    """The expected changes the agent hasn't made before ``turn``."""
    pending = list(expected)
    for i, msg in enumerate(view[:turn]):
        if msg.get("role") == "assistant":
            for call in changes_in(view, i, mutating):
                hit = next((e for e in pending if same_call(e, call)), None)
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


def watch_requests() -> None:
    """Print every teacher request OpenRouter doesn't answer cleanly (the client retries those
    quietly, with back-off), so a slow run shows why."""
    import httpx

    real = httpx.post

    def post(*args: Any, **kw: Any) -> Any:
        start = time.monotonic()
        try:
            r = real(*args, **kw)
        except httpx.HTTPError as e:
            print(
                f"   ↻ request failed after {time.monotonic() - start:.0f}s: {type(e).__name__}: {e}"[:200],
                flush=True,
            )
            raise
        if r.status_code != 200 or '"error"' in r.text[:2000]:
            print(
                f"   ↻ HTTP {r.status_code} after {time.monotonic() - start:.0f}s: {r.text[:160]!r}",
                flush=True,
            )
        return r

    httpx.post = post  # ty: ignore[invalid-assignment]  # a logging wrapper with the same call signature


def teach(
    failure: Any, point: FixPoint, mutating: Callable[[str], bool], attempts: int
) -> tuple[dict | None, list]:
    """The teacher's turn at ``point`` (`heal.PrefixTeacher`, plus `verify`): the accepted example
    as an on-policy one, and every try. Raises `runner.OutOfCredits` / ``ModelUnavailable``."""
    from reflexive import heal

    teacher = heal.PrefixTeacher(attempts=attempts, extra_check=lambda fixed: verify(fixed, point, mutating))
    try:
        ex = teacher.fix(failure)
    except ValueError as e:  # e.g. to_sft_example: the fixed turn didn't come out last
        log = teacher.tried.get(failure.task, [])
        return None, [*log, {"accepted": False, "problems": [str(e)]}]
    log = teacher.tried.get(failure.task, [])
    if ex is None:
        return None, log
    domain, task_id = failure.task
    sample = failure.record.get("sample", 0)
    src = ex["source"]
    ex["id"] = f"onpolicy/{src['teacher']}/{domain}/{task_id}/s{sample}/t{point.turn}/a{src['attempt']}"
    ex["group"] = "onpolicy"
    src.update(kind="onpolicy_fix", fix_kind=point.kind, fix_detail=point.detail, sample=sample)
    return ex, log


def val_ids(paths: Iterable[str]) -> tuple[set[tuple[str, str]], list[str]]:
    """The validation tasks in ``paths``, and why they can't be trusted to be all of them (empty if
    they can): each file read whole (no cut-short last line), and every domain's tasks present."""
    ids: set[tuple[str, str]] = set()
    problems = []
    for p in paths:
        try:
            ids |= {(r["domain"], str(r["task_id"])) for r in read_jsonl(Path(p))}
        except (BadJsonLine, FileNotFoundError) as e:
            problems.append(str(e))
    for d, n in VAL_TASKS.items():
        got = sum(1 for dom, _ in ids if dom == d)
        if got != n:
            problems.append(f"{d}: {got} validation tasks in --exclude, expected {n}")
    return ids, problems


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

    from reflexive.agent import Failure
    from reflexive.metrics import is_broken
    from reflexive.runner import ModelUnavailable, OutOfCredits
    from reflexive.splits import split_of

    logger.remove()
    if not a.exclude:
        print("⛔ no validation files to exclude (--exclude): refusing to risk fixing validation tasks")
        return 1
    banned, problems = val_ids(a.exclude)
    if problems:
        print("⛔ the validation exclusion isn't complete, refusing to risk fixing validation tasks:")
        for why in problems:
            print(f"   {why}")
        return 1
    if not a.dry_run:
        watch_requests()
    report: dict[str, Any] = {
        "files": [shown(f) for f in a.files],
        "excluded": len(banned),
        "conversations": [],
    }
    kinds: collections.Counter = collections.Counter()
    fixes: list[dict] = []
    out = None
    if not a.dry_run:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out = a.out.open("w")
    domains = Domains()
    try:
        for f in a.files:
            for r in latest_records(read_jsonl(f, partial_last_ok=True)):
                d, t = r["domain"], str(r["task_id"])
                if (r.get("outcome") or {}).get("passed") or is_broken(r):
                    continue
                if (d, t) in banned or split_of(d, t) != "train":
                    kinds["left out: a validation or non-train task"] += 1
                    continue
                point, diag, mutating = domains.fix_point(r)
                entry: dict[str, Any] = {"domain": d, "task_id": t, "sample": r.get("sample", 0)}
                report["conversations"].append(entry)
                if point is None:
                    kinds["no fix point (actions match; failed another check)"] += 1
                    entry["fix_point"] = None
                    continue
                kinds[point.kind] += 1
                entry["fix_point"] = {"turn": point.turn, "kind": point.kind, "detail": point.detail}
                if out is None:
                    continue
                done = sum(kinds[k] for k in ("unexpected_change", "change_never_made", "classifier"))
                start = time.monotonic()
                ex, entry["attempts"] = teach(
                    Failure(r, diagnosis_for(point, diag)), point, mutating, a.attempts
                )
                if ex is not None:
                    fixes.append(ex)
                    out.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    out.flush()
                tries = len(entry["attempts"])
                verdict = f"accepted on attempt {tries}" if ex is not None else f"refused ({tries} attempts)"
                print(
                    f"[{done}] {d} {t[:40]} {point.kind} → {verdict}, "
                    f"{time.monotonic() - start:.0f}s; {len(fixes)} accepted so far",
                    flush=True,
                )
    except (OutOfCredits, ModelUnavailable) as e:
        print(f"⛔ {type(e).__name__}: {e}"[:300] + "\n   stopping: what was accepted so far is saved")
        finish(a.out, out, report, kinds, fixes, f"{type(e).__name__}: {e}"[:300])
        return 1
    except BaseException as e:  # a crash or Ctrl-C: keep what was paid for, then let it through
        finish(a.out, out, report, kinds, fixes, f"{type(e).__name__}: {e}"[:300])
        raise
    finish(a.out, out, report, kinds, fixes)
    return 0


class Domains:
    """Each domain's τ²-bench tasks and tools, loaded once; the fix point of a failed record."""

    def __init__(self) -> None:
        self.tasks: dict[str, Any] = {}
        self.tools: dict[str, Any] = {}

    def fix_point(self, r: dict) -> tuple[FixPoint | None, Diagnosis, Callable[[str], bool]]:
        from tau2.registry import registry

        from reflexive.hallucination import agent_view_of, classify_record
        from reflexive.outcome import scorer

        d = r["domain"]
        if d not in self.tasks:
            self.tasks[d] = scorer(d)
            self.tools[d] = registry.get_env_constructor(d)(**(self.tasks[d].env_kwargs or {})).tools
        tools = self.tools[d]

        def mutating(name: str) -> bool:
            return tools.has_tool(name) and bool(tools.tool_mutates_state(name))

        ec = self.tasks[d].task(str(r["task_id"])).evaluation_criteria
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
        return point, diag, mutating


def finish(
    path: Path,
    out: Any,
    report: dict,
    kinds: collections.Counter,
    fixes: list[dict],
    stopped: str | None = None,
) -> None:
    """Close the fixes file and write the report beside it (nothing on a dry run)."""
    report["fix_points"] = dict(kinds)
    report["fixes"] = len(fixes)
    report["fixes_by_kind"] = dict(collections.Counter(x["source"]["fix_kind"] for x in fixes))
    report["stopped"] = stopped
    print(f"fix points: {dict(kinds)}")
    if out is None:
        print("(dry run: no teacher calls, nothing written)")
        return
    out.close()
    path.with_suffix(".report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(f"accepted fixes: {len(fixes)} {report['fixes_by_kind']} → {shown(path)}")


def diagnosis_for(point: FixPoint, diag: Diagnosis) -> Diagnosis:
    """The diagnosis `heal.fix_point` reads: the classifier's, or one finding at ``point``."""
    classifier = point.kind == "classifier" and diag.first is not None
    return Diagnosis(
        diag.failure if classifier else FailureType.NONE,
        [
            Finding(
                diag.first.type if classifier else FailureType.NONE,
                point.kind,
                point.turn,
                detail=point.detail,
            )
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
