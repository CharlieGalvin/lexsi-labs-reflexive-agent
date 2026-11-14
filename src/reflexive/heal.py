"""The real heal parts for `reflexive.agent.ReflexiveAgent` (SEL-47): the teacher's runtime fix,
a trainer that hands the fixes to a training run, and the audit record each cycle writes.

    teacher = PrefixTeacher()                        # GLM-5.3-Flash through OpenRouter
    trainer = CommandTrainer("runs/heal", command=["just", "sft-train", "{config}"])
    agent = ReflexiveAgent(rollout, teacher=teacher, trainer=trainer, gate=DevGate.load())
    report = agent.run_cycle(heal_tasks, gate_tasks)
    write_audit(report, "results/heal/audit.jsonl", cycle=1, teacher=teacher)

**The fix happens at runtime, from what the agent saw.** `scripts/teacher/fixes.py` re-runs the
whole τ²-bench task, which needs an environment that can be replayed; a live conversation
can't be. So the teacher gets the conversation exactly as the student saw it (``agent_view``),
**up to the failing turn** (the classifier's first finding), and writes that one turn again.
Nothing is re-run: no customer, no environment, no scorer.

**How a fix is checked without a scorer** (`check_fix`). AgentTune's ``ReplayValidator`` doesn't
help here: it runs a shell command and passes if it exits 0 (by default ``sys.exit(0)``).
Instead a fix is accepted only if:

1. it is a real turn (a message or at least one tool call), and differs from the failed one;
2. it doesn't give up: no ``###STOP###``, no ``transfer_to_human_agents`` hand-off, and a
   text-only turn says something (at least ``MIN_TEXT`` characters). A turn that does nothing
   can't hallucinate, so without this the cycle could train the student to stop rather than
   repair. Asking the customer for what's missing (every fix so far) is still accepted;
3. our classifier, run on the prefix plus the fixed turn, finds **nothing in the fixed turn**:
   no unknown or customer-only tool, no invented or ill-typed argument, no repeated failed call.

That is weaker than a scorer: it says the turn is not a Tool / Arg / Loop failure, not that the
task would now pass. The gate (`reflexive.gate.DevGate`) is what checks the adapter trained on
these fixes actually does better on dev. A replay check (re-run the prefix's tool calls on a
fresh τ²-bench database and check the fixed call returns no error) would be the next step up.

**What the trainer gets.** Each fix is a ``reagent.sft/v1`` example (the teammate's
`reflexive.train_data` format): the prefix and the fixed turn in Aya's own ``Action:`` format,
built by `teacher.to_prompted` as golden records are, with ``train_on`` = **the fixed turn only**.
The agent's earlier turns came from a conversation that failed, so they are context, never targets.

Stdlib only at import; the teacher model is made on first use.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reflexive.agent import Adapter, CycleReport, Failure, Record, Task
from reflexive.hallucination import agent_view_of, classify_hallucination, domain_tools
from reflexive.splits import split_of
from reflexive.teacher import to_prompted

SFT_FORMAT = "reagent.sft/v1"  # as `reflexive.train_data.FORMAT` (SEL-66)
AUDIT_FORMAT = "reagent.heal.audit/v1"
TEACHER = "glm53f"  # the bake-off's choice (SEL-58): z-ai/glm-5.3-flash
TEMPERATURES = (0.0, 0.7, 0.7)  # as the fix run: attempt 1 greedy, retries differ
STOP = "###STOP###"  # τ²-bench's end-of-conversation marker
HANDOFF = "transfer_to_human_agents"
MIN_TEXT = 20  # a text-only fix shorter than this ("OK.", "Sure.") does nothing

Message = dict[str, Any]
TurnAgent = Any  # anything with .turn(messages, tools) -> an OpenAI-style assistant message


# --- Where to fix ----------------------------------------------------------------------------


def fix_point(failure: Failure) -> tuple[list[Message], Message]:
    """The agent's view up to the failing turn (system prompt first), and the failing turn itself."""
    first = failure.diagnosis.first
    if first is None:
        raise ValueError(f"{failure.task}: the diagnosis has no finding, so no turn to fix")
    view = agent_view_of(failure.record)
    if not 0 < first.turn < len(view) or view[first.turn].get("role") != "assistant":
        raise ValueError(f"{failure.task}: turn {first.turn} isn't an agent turn of the saved view")
    if view[0].get("role") != "system":
        raise ValueError(f"{failure.task}: the saved view doesn't start with the system prompt")
    return view[: first.turn], view[first.turn]


def _clean_turn(msg: Message) -> Message:
    """The parts of an assistant message that matter for the conversation (no provider extras)."""
    out: Message = {"role": "assistant", "content": msg.get("content") or ""}
    calls = [
        {
            "id": tc.get("id") or f"fix_{i}",
            "type": "function",
            "function": {
                "name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments") or "{}",
            },
        }
        for i, tc in enumerate(msg.get("tool_calls") or [])
    ]
    if calls:
        out["tool_calls"] = calls
    return out


def _same_turn(a: Message, b: Message) -> bool:
    def key(m: Message) -> tuple:
        calls = []
        for tc in m.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = raw
            calls.append((tc["function"]["name"], json.dumps(args, sort_keys=True)))
        return ((m.get("content") or "").strip(), tuple(calls))

    return key(a) == key(b)


def check_fix(domain: str, prefix: list[Message], fixed: Message, failed: Message | None = None) -> list[str]:
    """Why ``fixed`` isn't an acceptable fix after ``prefix`` (empty if it is). See the module doc."""
    problems = []
    if not fixed.get("tool_calls") and not (fixed.get("content") or "").strip():
        problems.append("the fixed turn is empty")
    if failed is not None and _same_turn(fixed, failed):
        problems.append("the fixed turn is the failed turn again")
    text = (fixed.get("content") or "").strip()
    calls = [tc["function"]["name"] for tc in fixed.get("tool_calls") or []]
    if STOP in text:
        problems.append("the fixed turn ends the conversation")
    if HANDOFF in calls:
        problems.append("the fixed turn hands off to a human")
    if not calls and text and len(text) < MIN_TEXT:
        problems.append(f"the fixed turn is text only and too short to do anything: {text!r}")
    tools, customer = domain_tools(domain)
    at = len(prefix)
    found = classify_hallucination([*prefix, fixed], tools, customer_tools=customer).findings
    problems += [
        f"{f.type.value}/{f.kind} in the fixed turn: {f.detail or f.tool}" for f in found if f.turn == at
    ]
    return problems


def to_sft_example(failure: Failure, prefix: list[Message], fixed: Message, meta: dict[str, Any]) -> Record:
    """The fix as a ``reagent.sft/v1`` example: Aya's format, the fixed turn the only target."""
    domain, task_id = failure.task
    tools, _ = domain_tools(domain)
    msgs = [
        {"role": m["role"], "content": m["content"]} for m in to_prompted([*prefix, fixed], tools, "cohere")
    ]
    if msgs[-1]["role"] != "assistant":
        raise ValueError(f"{failure.task}: the fixed turn didn't come out last")
    first = failure.diagnosis.first
    assert first is not None  # fix_point checked
    return {
        "id": f"heal/{meta['teacher']}/{domain}/{task_id}/t{first.turn}/a{meta['attempt']}",
        "format": SFT_FORMAT,
        "domain": domain,
        "task_id": task_id,
        "group": "heal",
        "labels": [failure.diagnosis.failure.value],
        "messages": msgs,
        "train_on": [len(msgs) - 1],
        "source": {
            "kind": "runtime_fix",
            "adapter": failure.record.get("adapter"),
            "fixed_turn": first.turn,
            "finding": first.to_dict(),
            **meta,
        },
    }


# --- The teacher -----------------------------------------------------------------------------


@dataclass
class PrefixTeacher:
    """Fixes a failure at the failing turn: the teacher model sees the agent's view up to that turn
    and writes the turn again, up to ``attempts`` times, until `check_fix` accepts one.

    ``make_agent(temperature)`` builds the teacher; by default GLM-5.3-Flash via
    `reflexive.runner`. Only train tasks are fixed. ``tried`` keeps every attempt per task (what
    it wrote, why it was refused, cost) for the audit record.

    ``extra_check(fixed)``: more reasons to refuse a turn, on top of `check_fix` (e.g. the
    on-policy fixes' rule that every change must be one the task still expects). Running out of
    credits or a model that can't be reached (`reflexive.runner.OutOfCredits`, ``ModelUnavailable``)
    stops the run: every later attempt would fail too, so they aren't logged as refusals.
    """

    make_agent: Callable[[float], TurnAgent] | None = None
    model: str = TEACHER
    attempts: int = 3
    extra_check: Callable[[Message], list[str]] | None = None
    tried: dict[Task, list[dict[str, Any]]] = field(default_factory=dict)

    def _agent(self, temperature: float) -> TurnAgent:
        if self.make_agent is not None:
            return self.make_agent(temperature)
        from reflexive import runner

        agent = runner.make_agent(self.model)()
        if isinstance(agent, runner.NativeAgent):
            agent.temperature = temperature
        return agent

    def fix(self, failure: Failure) -> Record | None:
        domain, task_id = failure.task
        if split_of(domain, task_id) != "train":
            raise ValueError(f"{domain} task {task_id!r} isn't a train task: fixes are only made on train")
        log = self.tried.setdefault(failure.task, [])
        try:
            prefix, failed = fix_point(failure)
        except ValueError as e:
            log.append({"attempt": 0, "accepted": False, "problems": [str(e)]})
            return None
        tools, _ = domain_tools(domain)
        for i in range(1, self.attempts + 1):
            temperature = TEMPERATURES[min(i, len(TEMPERATURES)) - 1]
            agent = self._agent(temperature)
            start = time.monotonic()
            try:
                fixed = _clean_turn(agent.turn([dict(m) for m in prefix], tools))
            except Exception as e:  # noqa: BLE001 - an API failure is a refused attempt, not a crash
                from reflexive.runner import ModelUnavailable, OutOfCredits

                if isinstance(e, (OutOfCredits, ModelUnavailable)):
                    raise
                log.append(
                    {
                        "attempt": i,
                        "temperature": temperature,
                        "accepted": False,
                        "error": f"{type(e).__name__}: {e}"[:300],
                    }
                )
                continue
            problems = check_fix(domain, prefix, fixed, failed)
            if self.extra_check is not None:
                problems += self.extra_check(fixed)
            usage = dict(getattr(agent, "usage", {}) or {})
            log.append(
                {
                    "attempt": i,
                    "temperature": temperature,
                    "accepted": not problems,
                    "problems": problems,
                    "turn": fixed,
                    "seconds": round(time.monotonic() - start, 2),
                    "usage": usage,
                }
            )
            if not problems:
                meta = {
                    "teacher": self.model,
                    "attempt": i,
                    "temperature": temperature,
                    "failed_turn": _clean_turn(failed),
                    "fixed_turn_message": fixed,
                }
                return to_sft_example(failure, prefix, fixed, meta)
        return None


# --- The trainer ------------------------------------------------------------------------------


@dataclass
class CommandTrainer:
    """Writes the fixes as a ``reagent.sft/v1`` JSONL file and runs a training command on them.

    ``command`` is a list of arguments; ``{data}``, ``{base}``, ``{name}`` and ``{out}`` in it are
    replaced by the fixes file, the base adapter (empty for the base student), the new adapter's
    name and its output directory. The new adapter's id is ``name`` (the trained adapter is
    expected in ``out``). With no command (``dry_run``), the data is written and nothing trains:
    the adapter id is still returned, so a cycle can be run against a rollout that knows it.

    The real command is the teammate's SFT run (`scripts/sft/train_sft.py`, SEL-65, on the GPU pod).
    """

    root: Path | str = "runs/heal"
    command: Sequence[str] | None = None
    prefix: str = "heal"
    per_lesson: int | None = 2  # most fixes kept per `lesson` (None: keep all)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def train(self, fixes: Sequence[Record], base: Adapter) -> Adapter:
        if not fixes:
            raise ValueError("nothing to train on")
        bad = [f.get("id") for f in fixes if f.get("format") != SFT_FORMAT]
        if bad:
            raise ValueError(f"not {SFT_FORMAT} examples: {bad[:5]}")
        fixes, dropped = dedupe(fixes, self.per_lesson)
        n = len(self.calls) + 1
        name = f"{self.prefix}-{time.strftime('%Y%m%d-%H%M%S')}-c{n}"
        out = Path(self.root) / name
        out.mkdir(parents=True, exist_ok=False)
        data = out / "fixes.jsonl"
        data.write_text("".join(json.dumps(f) + "\n" for f in fixes))
        call: dict[str, Any] = {
            "name": name,
            "base": base,
            "data": str(data),
            "examples": len(fixes),
            "per_lesson": self.per_lesson,
            "dropped_duplicates": [f.get("id") for f in dropped],
            "lessons": dict(Counter(lesson(f) for f in fixes)),
        }
        if self.command:
            subs = {"data": str(data), "base": base or "", "name": name, "out": str(out)}
            argv = [_fill(a, subs) for a in self.command]
            call["argv"] = argv
            start = time.monotonic()
            done = subprocess.run(argv, check=False, capture_output=True, text=True)  # noqa: S603 - our own config
            call.update(returncode=done.returncode, seconds=round(time.monotonic() - start, 1))
            (out / "train.log").write_text(done.stdout + done.stderr)
            if done.returncode != 0:
                self.calls.append(call)
                raise RuntimeError(f"training failed (exit {done.returncode}); see {out / 'train.log'}")
        self.calls.append(call)
        return name


def lesson(example: Record) -> str:
    """What a fix teaches: the mistake it corrects and what the fixed turn does instead.

    ``ARG/placeholder/get_customer_by_phone -> message`` is "don't invent the phone number: ask
    for it", whichever conversation it came from. Fixes with the same lesson are near-duplicates.
    """
    src = example.get("source") or {}
    f = src.get("finding") or {}
    turn = src.get("fixed_turn_message") or {}
    calls = sorted({tc["function"]["name"] for tc in turn.get("tool_calls") or []})
    does = "+".join(calls) if calls else "message"
    return f"{example.get('domain')}:{f.get('type')}/{f.get('kind')}/{f.get('tool')} -> {does}"


def dedupe(fixes: Sequence[Record], per_lesson: int | None) -> tuple[list[Record], list[Record]]:
    """At most ``per_lesson`` fixes per `lesson`, the first ones in order; and the ones dropped."""
    if per_lesson is None:
        return list(fixes), []
    if per_lesson < 1:
        raise ValueError("per_lesson must be at least 1 (or None to keep every fix)")
    seen: Counter[str] = Counter()
    kept, dropped = [], []
    for f in fixes:
        key = lesson(f)
        seen[key] += 1
        (kept if seen[key] <= per_lesson else dropped).append(f)
    return kept, dropped


def _fill(arg: str, subs: dict[str, str]) -> str:
    """Only the known ``{placeholders}``: other braces (JSON, inline code) are left as they are."""
    for k, v in subs.items():
        arg = arg.replace("{" + k + "}", v)
    return arg


# --- The audit record -------------------------------------------------------------------------


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def audit_record(report: CycleReport, *, cycle: int, teacher: Any = None, trainer: Any = None) -> Record:
    """One cycle, as a JSON-ready audit record: failures, fixes accepted, gate decision, adapter."""
    tried = getattr(teacher, "tried", {}) or {}
    fixed = {(f["domain"], str(f["task_id"])) for f in report.fixes}
    failures = []
    for f in report.failures:
        first = f.diagnosis.first
        attempts = tried.get(f.task, [])
        failures.append(
            {
                "domain": f.task[0],
                "task_id": f.task[1],
                "failure": f.diagnosis.failure.value,
                "first": first.to_dict() if first else None,
                "kinds": dict(f.diagnosis.kinds),
                "fixed": f.task in fixed,
                "attempts": [{k: v for k, v in a.items() if k != "turn"} for a in attempts],
            }
        )
    train = (getattr(trainer, "calls", None) or [None])[-1] if report.candidate else None
    return {
        "format": AUDIT_FORMAT,
        "cycle": cycle,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": _git_sha(),
        "adapter": {
            "before": report.adapter_before,
            "candidate": report.candidate,
            "after": report.adapter_after,
            "swapped": report.swapped,
        },
        "heal": {
            "tasks": [[r["domain"], str(r["task_id"])] for r in report.rollouts],
            "passed": sum(bool((r.get("outcome") or {}).get("passed")) for r in report.rollouts),
            "broken": [[r["domain"], str(r["task_id"])] for r in report.broken],
        },
        "failures": failures,
        "fixes": {
            "accepted": len(report.fixes),
            "of": len(report.failures),
            "ids": [f.get("id") for f in report.fixes],
            "lessons": dict(Counter(lesson(f) for f in report.fixes)),
        },
        "training": train,
        "gate": None if report.gate is None else _gate_json(report),
        "skipped": report.skipped,
    }


def _gate_json(report: CycleReport) -> dict[str, Any]:
    g = report.gate
    assert g is not None
    out = json.loads(json.dumps(vars(g), default=str))
    out["dropped"] = [list(t) for t in report.gate_dropped]
    out["tasks"] = [[r["domain"], str(r["task_id"])] for r in report.gate_baseline]
    return out


def write_audit(report: CycleReport, path: Path | str, **kw: Any) -> Record:
    """Append this cycle's audit record to ``path`` (JSONL) and return it."""
    rec = audit_record(report, **kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


# --- A run's metrics (the committed evidence) ---------------------------------------------------

METRICS_FORMAT = "reagent.heal.metrics/v1"


def run_metrics(audits: Sequence[Record], episodes: Sequence[Record], run: Record | None = None) -> Record:
    """One heal run summed up from its audit records and conversations: small enough to commit
    as ``results/heal-<name>.metrics.json`` (raw ``results/`` files are gitignored).

    Costs are the student's and the customer's (from each conversation) and the teacher's (from
    each attempt). τ²-bench's judge calls aren't priced in the records, so they're not included.
    """
    from reflexive.stats import wilson

    heal = [e for e in episodes if e.get("heal_role", "heal") == "heal"]
    by_task = {(e["domain"], str(e["task_id"]), e.get("cycle", 1)): e for e in heal}
    cycles = []
    for a in audits:
        n, k = len(a["heal"]["tasks"]), a["heal"]["passed"]
        lo, hi = wilson(k, n)
        fails = a["failures"]
        fixable = [f for f in fails if f["first"] is not None]
        tasks = []
        for f in fails:
            e = by_task.get((f["domain"], str(f["task_id"]), a["cycle"])) or {}
            o = e.get("outcome") or {}
            tasks.append(
                {
                    "domain": f["domain"],
                    "task_id": f["task_id"],
                    "termination": e.get("termination"),
                    "checks": o.get("checks"),
                    "failure": f["failure"],
                    "first": None
                    if f["first"] is None
                    else {k: f["first"].get(k) for k in ("type", "kind", "tool", "turn", "detail")},
                    "fixed": f["fixed"],
                    "attempts": len([t for t in f["attempts"] if t.get("attempt")]),
                }
            )
        types: Counter[str] = Counter(f["failure"] for f in fails)
        cycles.append(
            {
                "cycle": a["cycle"],
                "adapter": a["adapter"],
                "skipped": a["skipped"],
                "conversations": n,
                "passed": k,
                "task_success": round(k / n, 4) if n else None,
                "task_success_wilson95": [round(lo, 4), round(hi, 4)],
                "broken": len(a["heal"]["broken"]),
                "failures": len(fails),
                "failure_types": {t: types.get(t, 0) for t in ("TOOL", "ARG", "LOOP", "NONE")},
                "fixable": len(fixable),
                "fixed": sum(f["fixed"] for f in fails),
                "first_try": sum(
                    f["fixed"] and len([t for t in f["attempts"] if t.get("attempt")]) == 1 for f in fails
                ),
                "fixed_at_turn": dict(Counter(str(f["first"]["turn"]) for f in fixable if f["fixed"])),
                "lessons": a["fixes"].get("lessons", {}),
                "training": a.get("training"),
                "gate": a.get("gate"),
                "tasks": tasks,
            }
        )
    student = sum(((e.get("agent_usage") or {}).get("est_cost") or 0) for e in episodes)
    customer = sum((e.get("user_cost") or 0) for e in episodes)
    teacher = sum(
        ((t.get("usage") or {}).get("billed_cost") or 0)
        for a in audits
        for f in a["failures"]
        for t in f["attempts"]
    )
    return {
        "format": METRICS_FORMAT,
        "run": run or {},
        "conversations": len(episodes),
        "cycles": cycles,
        "cost_usd": {
            "student": round(student, 4),
            "customer": round(customer, 4),
            "teacher": round(teacher, 4),
            "total": round(student + customer + teacher, 4),
            "note": "judge calls not included",
        },
    }
