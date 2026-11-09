"""The runtime monitor (SEL-62): find failures while the conversation runs, not after it.

    mon = RuntimeMonitor(tools=ep.tool_schemas)       # the live request's tools, no dataset rules
    for each turn:
        new = mon.observe(ep.messages())              # the agent's view so far
        for alarm in new: ...                         # log it, nudge, or stop (`Guard`)

**What it checks**, after every turn, on the conversation as the agent saw it:

* ``no_progress`` (Loop, new): **3 agent turns in a row with no tool call, each repeating
  something the agent already said.** Text is compared after lowercasing and keeping letters
  and digits only (so two different order numbers are different text). A tool call written as
  text (a prompted model's ``Action:`` block, before the harness parses it) counts as a call.
  This catches the loops AgentTune's ``FailureDetector`` can't see, because they have no tool
  calls: the agent and the customer swapping goodbyes until τ²-bench's step limit. On all
  148 Arm A and C telecom conversations it fires on all 7 that ran to the limit (A56 at agent
  turn 44 of 100) and on none of the 7 that passed.
* the classifier's Tool / Arg / Loop findings (`hallucination.classify_hallucination`), on the
  growing conversation, with the tools **from the live request**: the monitor knows no dataset.
  Customer-only tool names are an optional extra label.

Each finding is reported once, on the turn it first appears. ``no_progress`` measures wasted
turns, not failure: a conversation that passed can end in a goodbye loop too. So `MonitorDetector`
only hands ``FullClosedLoop`` the alarms of conversations that **didn't pass**, read from the
outcome line the harness writes at the end (`Watcher.finish`). An `Alarm` converts to AgentTune's
``Failure`` (`to_failure`), and a conversation to AgentTune's ``EventLog`` (`event_log`) and to
the audit-log lines ``FullClosedLoop`` reads (`audit_records`); `MonitorDetector` stands in for
``FullClosedLoop.detector``. AgentTune itself isn't changed, and is imported only by those
adapters: the rest is stdlib.

**What happens when it fires** (`Guard`): in ``log`` mode, nothing but the record (the
benchmark arms: a guard that changes the conversation is measured as its own ablation, so Arm
B's gain isn't confused with it). In ``act`` mode, the first ``no_progress`` adds a nudge to
the agent's next prompt; if the agent keeps looping (another ``no_progress`` after the nudge),
the conversation is stopped. Other findings are only logged.

**Checked against AgentTune itself** (7 Nov, the 148 Arm A and C telecom conversations written
as audit logs after every message): its own ``FailureDetector`` reads our lines with no schema
errors, and our text and observation lines never trigger its ``loop_collapse``. It flags
``tool_crash`` on all 148, passes included, and its ``loop_collapse`` finds none of the 7 loops,
which is why the monitor stands in for it. A real ``FullClosedLoop`` with `plug_into` runs
``ingest_once`` through our detector and classifier (tested). Its own ``DeploymentGate``
builds its test set from DECIDE pipeline records (``pipeline_id`` and a completion verdict),
which a conversation's audit log doesn't have, so on its own it would approve every adapter:
our gate is `reflexive.gate.DevGate` (SEL-51).

Known gap: a loop that rewords itself each time gets past the exact comparison. Comparing by
similarity is the follow-up, at the risk of more false alarms.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from reflexive.hallucination import ACTION_TEXT, TO_AGENTTUNE, FailureType, classify_hallucination

Message = dict[str, Any]
NO_PROGRESS_RUN = 3  # agent turns in a row
NUDGE = (
    "Note from the system, not the customer: you have repeated yourself several times. "
    "If the task is done, end the conversation as your instructions say."
)
MODES = ("off", "log", "act")


def normalise(text: str | None) -> str:
    """Lowercase, letters and digits only: "Thank you!" and "thank you." are the same, but
    "Order #W2378156 is cancelled" and "Order #W9911002 is cancelled" aren't."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def is_call(m: Message) -> bool:
    """A turn that calls a tool: parsed ``tool_calls``, or a prompted ``Action:`` block still in text."""
    return bool(m.get("tool_calls")) or bool(ACTION_TEXT.search(m.get("content") or ""))


@dataclass(frozen=True)
class Alarm:
    """One failure the monitor found, on the agent turn it first appeared in."""

    type: str  # FailureType value: TOOL, ARG or LOOP
    kind: str  # e.g. no_progress, customer_tool, ungrounded
    turn: int  # index of the agent's message in the view (as classifier findings)
    agent_turn: int  # 1 = the agent's first reply
    tool: str | None = None
    detail: str = ""
    call: int = 0  # position of the call within the turn, as classifier findings

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def no_progress_turns(view: Sequence[Message], run: int = NO_PROGRESS_RUN) -> list[tuple[int, int]]:
    """Each (message index, agent turn) where ``run`` text-only repeats in a row are reached.

    After firing, the count starts again, so a loop that carries on fires again ``run`` turns later.
    """
    said: set[str] = set()
    streak, agent_turn, out = 0, 0, []
    for i, m in enumerate(view):
        if m.get("role") != "assistant":
            continue
        agent_turn += 1
        text = normalise(m.get("content"))
        if not is_call(m) and text and text in said:
            streak += 1
            if streak == run:
                out.append((i, agent_turn))
                streak = 0
        else:
            streak = 0
        if text:
            said.add(text)
    return out


@dataclass
class RuntimeMonitor:
    """Checks the agent's growing view after each turn; `observe` returns what's new."""

    tools: list[dict]
    customer_tools: Sequence[str] = ()
    run: int = NO_PROGRESS_RUN
    alarms: list[Alarm] = field(default_factory=list)
    _seen: set[tuple] = field(default_factory=set)

    def observe(self, view: Sequence[Message]) -> list[Alarm]:
        agent_turns = {}
        n = 0
        for i, m in enumerate(view):
            if m.get("role") == "assistant":
                n += 1
                agent_turns[i] = n
        new: list[Alarm] = []
        for i, k in no_progress_turns(view, self.run):
            a = Alarm(
                FailureType.LOOP.value,
                "no_progress",
                i,
                k,
                None,
                f"{self.run} agent turns in a row with no tool call, each repeating what it already said",
            )
            new += self._add(a, ("no_progress", i))
        diag = classify_hallucination(list(view), self.tools, customer_tools=set(self.customer_tools))
        for f in diag.findings:
            a = Alarm(f.type.value, f.kind, f.turn, agent_turns.get(f.turn, 0), f.tool, f.detail, f.call)
            new += self._add(a, (f.kind, f.turn, f.call, f.tool))
        new.sort(key=lambda a: a.turn)
        self.alarms += new
        return new

    def _add(self, a: Alarm, key: tuple) -> list[Alarm]:
        if key in self._seen:
            return []
        self._seen.add(key)
        return [a]


# --- What to do when it fires ---------------------------------------------------------------


@dataclass
class Guard:
    """The response to ``no_progress``: ``off``, ``log`` (record only) or ``act`` (nudge, then stop)."""

    mode: str = "log"
    nudged_at: int | None = None  # agent turn of the nudge
    stopped_at: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"monitor mode must be one of {MODES}, not {self.mode!r}")

    def react(self, new: Sequence[Alarm]) -> str | None:
        """``"nudge"``, ``"stop"`` or None, for the alarms this turn raised."""
        if self.mode != "act" or self.stopped_at is not None:
            return None
        loop = next((a for a in new if a.kind == "no_progress"), None)
        if loop is None:
            return None
        if self.nudged_at is None:
            self.nudged_at = loop.agent_turn
            return "nudge"
        self.stopped_at = loop.agent_turn
        return "stop"

    @staticmethod
    def nudged(prompt: list[Message]) -> list[Message]:
        """The prompt with the nudge as a last user turn (a system turn would repeat the tool list)."""
        return [*prompt, {"role": "user", "content": NUDGE}]

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "nudged_at": self.nudged_at, "stopped_at": self.stopped_at}


# --- The audit log (what FullClosedLoop reads) ----------------------------------------------


def _bare(tc: dict) -> dict:
    """An OpenAI tool call as AgentTune's bare ``{"name", "arguments"}``."""
    fn = tc.get("function") or {}
    raw = fn.get("arguments") or {}
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        args = {"_raw": raw}
    return {
        "name": fn.get("name") or "unknown",
        "arguments": args if isinstance(args, dict) else {"_raw": args},
    }


def audit_records(view: Sequence[Message], trajectory_id: str, start: int = 0) -> list[dict]:
    """The view's messages from index ``start`` on, as audit-log lines.

    Tool calls are exactly the records AgentTune's ``EventLog.to_audit_records`` makes (``stage_name``
    = tool and arguments, ``status`` ``error`` with ``error_details`` when the result starts with
    "Error"), so its ``FailureDetector`` reads them unchanged. Everything else the monitor needs
    is added as its own stage types, each with a unique ``stage_name`` so that detector never
    counts them as revisits: ``observation`` (what the agent was shown) and ``agent_text`` (what
    it said). Each line keeps its ``message_index`` and the original message, so a view can be
    rebuilt from the log (`view_from_audit`).
    """
    results = {m.get("tool_call_id"): str(m.get("content") or "") for m in view if m.get("role") == "tool"}
    out: list[dict] = []
    for i in range(start, len(view)):
        m = view[i]
        role = m.get("role")
        base = {"trajectory_id": trajectory_id, "message_index": i, "message": m}
        if role == "assistant":
            turn: list[dict] = []
            if (m.get("content") or "").strip() or not m.get("tool_calls"):
                turn.append(
                    {
                        **base,
                        "stage_name": f"agent_text#{i}",
                        "stage_type": "agent_text",
                        "state_snapshot": {"text": m.get("content") or ""},
                        "status": "ok",
                    }  # fmt: skip
                )
            for tc in m.get("tool_calls") or []:
                action = _bare(tc)
                name, args = action["name"], action["arguments"]
                stage = f"{name} {json.dumps(args, sort_keys=True, default=str)}" if args else name
                rec = {
                    **base,
                    "stage_name": stage,
                    "tool_name": name,
                    "stage_type": "tool_call",
                    "state_snapshot": {"arguments": args},
                    "status": "ok",
                }
                res = results.get(tc.get("id"))
                if res is not None and res.startswith("Error"):
                    rec.update(status="error", error_details=res[:500])
                turn.append(rec)
            for r in turn[1:]:  # the message is kept once, on the turn's first line
                r["message"] = None
            out += turn
        else:
            out.append(
                {
                    **base,
                    "stage_name": f"{role}#{i}",
                    "stage_type": "observation",
                    "state_snapshot": {"role": role},
                    "status": "ok",
                }  # fmt: skip
            )
    return out


def view_from_audit(records: Sequence[dict]) -> dict[str, list[Message]]:
    """Each trajectory's view, rebuilt from its audit lines (in message order)."""
    by: dict[str, dict[int, Message]] = {}
    for r in records:
        if r.get("message") is not None and "message_index" in r:
            by.setdefault(r["trajectory_id"], {})[r["message_index"]] = r["message"]
    return {t: [msgs[i] for i in sorted(msgs)] for t, msgs in by.items()}


_WRITE_LOCK = threading.Lock()  # conversations run in threads may share one audit log


def _ready(view: Sequence[Message]) -> int:
    """How much of the view can be logged for good: all of it, unless its last agent turn still
    waits for tool results (a call's ``status`` depends on its result)."""
    answered = {m.get("tool_call_id") for m in view if m.get("role") == "tool"}
    for j in range(len(view) - 1, -1, -1):
        m = view[j]
        if m.get("role") == "assistant":
            pending = any(tc.get("id") not in answered for tc in m.get("tool_calls") or [])
            return j if pending else len(view)
    return len(view)


class AuditWriter:
    """Appends a conversation's new messages to a JSONL audit log after each turn.

    An agent turn whose tool results haven't come back yet is held until they have (or until
    `close`), so a failed call is always logged as ``error``.
    """

    def __init__(self, path: Path | str, trajectory_id: str) -> None:
        self.path, self.trajectory_id, self.done = Path(path), trajectory_id, 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, view: Sequence[Message], *, final: bool = False) -> int:
        upto = len(view) if final else _ready(view)
        if upto <= self.done:
            return 0
        recs = audit_records(view[:upto], self.trajectory_id, self.done)
        if recs:
            text = "".join(json.dumps(r, default=str) + "\n" for r in recs)
            with _WRITE_LOCK, self.path.open("a") as f:
                f.write(text)
        self.done = upto
        return len(recs)

    def close(self, view: Sequence[Message]) -> int:
        """Write what's left at the end of the conversation, results or not."""
        return self.write(view, final=True)

    def outcome(self, passed: bool, termination: str | None = None) -> None:
        """The conversation's last line: whether it passed (its own stage, as the text lines)."""
        line = {
            "trajectory_id": self.trajectory_id,
            "stage_name": "outcome",
            "stage_type": "outcome",
            "state_snapshot": {"passed": bool(passed), "termination": termination},
            "status": "ok",
        }
        with _WRITE_LOCK, self.path.open("a") as f:
            f.write(json.dumps(line) + "\n")


def outcomes_from_audit(records: Sequence[dict]) -> dict[str, bool]:
    """Each finished trajectory's pass, from its outcome line (a running one has none yet)."""
    return {
        r["trajectory_id"]: bool(r["state_snapshot"]["passed"])
        for r in records
        if r.get("stage_type") == "outcome"
    }


class Watcher:
    """What a harness runs after every turn: write the audit log, check, and decide the response.

    ``after_turn(view)`` returns ``"nudge"``, ``"stop"`` or None. A monitor error is kept in
    ``error`` and switches it off: it must never break the conversation it watches.
    """

    def __init__(
        self,
        mode: str,
        tools: list[dict],
        audit_log: Path | str | None = None,
        trajectory_id: str = "",
        customer_tools: Sequence[str] = (),
    ) -> None:
        self.guard = Guard(mode)
        self.monitor = RuntimeMonitor(tools, customer_tools) if mode != "off" else None
        self.writer = AuditWriter(audit_log, trajectory_id) if audit_log else None
        self.error: str | None = None

    def after_turn(self, view: Sequence[Message]) -> str | None:
        if self.monitor is None and self.writer is None:
            return None
        try:
            if self.writer is not None:
                self.writer.write(view)
            return self.guard.react(self.monitor.observe(view)) if self.monitor is not None else None
        except Exception as e:  # noqa: BLE001 - the monitor must never break a conversation
            self.error = f"{type(e).__name__}: {str(e)[:200]}"
            self.monitor = self.writer = None
            return None

    def close(self, view: Sequence[Message]) -> None:
        """At the end of the conversation: log whatever is still held back."""
        if self.writer is not None:
            try:
                self.writer.close(view)
            except Exception as e:  # noqa: BLE001 - never break the run over the log
                self.error = f"{type(e).__name__}: {str(e)[:200]}"

    def finish(self, passed: bool, termination: str | None = None) -> None:
        """Once the conversation is scored: log its outcome, so the detector knows whether it failed."""
        if self.writer is not None:
            try:
                self.writer.outcome(passed, termination)
            except Exception as e:  # noqa: BLE001 - never break the run over the log
                self.error = f"{type(e).__name__}: {str(e)[:200]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.guard.to_dict(),
            "alarms": [a.to_dict() for a in self.monitor.alarms] if self.monitor is not None else [],
            "error": self.error,
        }


# --- AgentTune adapters (import AgentTune only here) ----------------------------------------


def to_failure(alarm: Alarm, trajectory_id: str) -> Any:
    """The alarm as AgentTune's ``Failure``: our labels in its taxonomy (``loop_collapse`` …)."""
    from agenttune.decide.closed_loop.contracts import Failure

    root = "no_progress" if alarm.kind == "no_progress" else TO_AGENTTUNE[FailureType(alarm.type)]
    return Failure(
        trajectory_id=trajectory_id,
        failure_type=root,
        failed_stage_name=alarm.tool or f"agent_turn_{alarm.agent_turn}",
        context={**alarm.to_dict(), "monitor": "reflexive.monitor"},
        error_message=alarm.detail,
    )


def event_log(view: Sequence[Message], trajectory_id: str | None = None) -> Any:
    """The conversation as AgentTune's light-tier ``EventLog``: what was said, called and shown."""
    from agenttune.agentic.events import Event, EventKind, EventLog

    log = EventLog(tier="light", **({"id": trajectory_id} if trajectory_id else {}))
    results = {m.get("tool_call_id"): str(m.get("content") or "") for m in view if m.get("role") == "tool"}
    for i, m in enumerate(view):
        role = m.get("role")
        if role == "assistant":
            if (m.get("content") or "").strip():
                log.append(Event(EventKind.TEXT, {"text": m["content"], "message_index": i}))
            for tc in m.get("tool_calls") or []:
                log.append(Event(EventKind.TOOL_CALL, {"action": _bare(tc), "message_index": i}))
                res = results.get(tc.get("id"))
                if res is not None:
                    payload: dict[str, Any] = {"output": res}
                    if res.startswith("Error"):
                        payload["error"] = res[:500]
                    log.append(Event(EventKind.TOOL_RESULT, payload))
            log.append(Event(EventKind.TURN_COMPLETE, {"message_index": i}))
        elif role in ("user", "system"):
            log.append(Event(EventKind.OBSERVATION, {"text": m.get("content") or "", "role": role}))
    return log


def record_event_log(rec: dict) -> Any:
    """A saved episode record (``reagent.tau2.episode``) as an ``EventLog``: the converter."""
    from reflexive.hallucination import agent_view_of

    return event_log(agent_view_of(rec), f"{rec.get('domain')}/{rec.get('task_id')}/{rec.get('sample', 0)}")


class MonitorDetector:
    """Stands in for ``FullClosedLoop.detector``: ``scan_audit_log(path)`` yields AgentTune ``Failure``s.

    Reads the whole log each time, rebuilds every trajectory's view, runs the monitor, and yields
    only failures it hasn't yielded before (so it can be polled as the log grows).

    With ``failed_only`` (the default) it yields only for conversations whose outcome line says
    they didn't pass: a passing conversation's goodbye loop or customer-tool call isn't a failure
    to train on, and a conversation still running is held until its outcome is in.
    """

    def __init__(
        self,
        tools_for: Any,
        customer_tools_for: Any = None,
        run: int = NO_PROGRESS_RUN,
        failed_only: bool = True,
    ) -> None:
        # tools_for(trajectory_id) -> the tools that conversation's agent was given
        self.tools_for, self.customer_tools_for, self.run = tools_for, customer_tools_for, run
        self.failed_only = failed_only
        self._yielded: set[tuple] = set()

    def scan_audit_log(self, file_path: str) -> Iterator[Any]:
        path = Path(file_path)
        if not path.exists():
            return
        recs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        passed = outcomes_from_audit(recs)
        for traj, view in view_from_audit(recs).items():
            if self.failed_only and passed.get(traj, True):  # passed, or not finished yet
                continue
            customer = self.customer_tools_for(traj) if self.customer_tools_for else ()
            mon = RuntimeMonitor(self.tools_for(traj), customer, self.run)
            for a in mon.observe(view):
                key = (traj, a.kind, a.turn, a.call, a.tool)
                if key not in self._yielded:
                    self._yielded.add(key)
                    yield to_failure(a, traj)


# AgentTune's root causes: ours map onto them; no_progress is a loop, so loop_collapse.
_ROOT_CAUSE = {"no_progress": "loop_collapse", **{v: v for v in TO_AGENTTUNE.values()}}


class MonitorClassifier:
    """Stands in for ``FullClosedLoop.classifier``: the monitor's failures are already classified
    (deterministically), so ``classify_batch`` labels them without an LLM call."""

    async def classify_batch(self, failures: list[Any]) -> list[Any]:
        from agenttune.decide.closed_loop.contracts import ClassifiedFailure

        return [
            ClassifiedFailure(
                failure=f,
                root_cause=_ROOT_CAUSE.get(f.failure_type, "incomplete_reasoning"),
                confidence=1.0,
                analysis=f"monitor: {f.context.get('type')}/{f.context.get('kind')}: {f.error_message}",
            )
            for f in failures
        ]


def plug_into(loop: Any, tools_for: Any, customer_tools_for: Any = None, failed_only: bool = True) -> Any:
    """Replace a ``FullClosedLoop``'s detector and classifier with ours (AgentTune is unchanged)."""
    loop.detector = MonitorDetector(tools_for, customer_tools_for, failed_only=failed_only)
    loop.classifier = MonitorClassifier()
    return loop
