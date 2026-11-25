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
the conversation is stopped, with its own termination ``monitor_stop`` (SEL-69): τ²-bench
grades it as an agent stop, but metrics report those runs on their own. Other findings are only
logged.

**Before a change runs** (`ActionGuard`, SEL-89): the checks above come after a turn, which is
too late for a write. So each state-changing call the agent proposes (τ²-bench's ``WRITE``
tools) first goes through `reflexive.actions.check_action`, in the same mode. In ``act`` mode a
live finding blocks the reply before τ²-bench sees it, and the agent is asked again with
`BLOCK_MESSAGE` ("you haven't got the customer's yes"); the conversation carries on. Every
decision is recorded (``rec["monitor"]["actions"]``).

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
from collections.abc import Collection, Iterator, Sequence
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
    nudge: dict[str, Any] | None = None  # the nudge as the agent saw it (`nudged`)

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

    def nudged(self, prompt: list[Message]) -> list[Message]:
        """The prompt with the nudge as a last user turn (a system turn would repeat the tool list).

        The nudge goes into this one prompt only, never into the conversation τ²-bench keeps, so
        it is recorded here (`nudge`, SEL-69): its text, where it went in the agent's view (after
        message ``after_view_index``, whose role is ``after_role``) and the agent turn it was
        shown before. After a customer message, the agent sees two user turns in a row."""
        last = prompt[-1] if prompt else {}
        self.nudge = {
            "text": NUDGE,
            "role": "user",
            "after_view_index": len(prompt) - 1,
            "after_role": last.get("role"),
            "before_agent_turn": (self.nudged_at or 0) + 1,
        }
        return [*prompt, {"role": "user", "content": NUDGE}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "nudged_at": self.nudged_at,
            "stopped_at": self.stopped_at,
            "nudge": self.nudge,
        }


# --- Before a change runs (SEL-89) -----------------------------------------------------------

BLOCK_MESSAGE = (
    "Note from the system, not the customer: you haven't got the customer's yes. "
    "Describe the change and ask them to confirm."
)
MAX_BLOCKS_PER_CHANGE = 2  # the same change proposed a third time runs, recorded as "overridden"
MAX_BLOCKS = 10  # blocks per conversation: a re-ask takes no τ²-bench step, so it needs its own limit


def _change_key(call: dict) -> str:
    """The same tool with the same arguments is the same change, however the JSON was spelled."""
    bare = _bare(call)
    return f"{bare['name']} {json.dumps(bare['arguments'], sort_keys=True, default=str)}"


@dataclass
class ActionGuard:
    """Checks each state-changing call the agent proposes, **before** τ²-bench runs it (SEL-89).

    `check` runs `reflexive.actions.check_action` on every call in a reply whose tool is in
    ``write_tools`` (τ²-bench's ``ToolType.WRITE`` tools, read from the live environment). What
    it does with a finding depends on ``mode``, as for `Guard`:

    * ``off``: nothing is checked.
    * ``log``: every finding is recorded (``action="logged"``) and the call runs.
    * ``act``: a **live** finding blocks the whole reply. It isn't sent to τ²-bench, and the agent
      is asked again with `BLOCK_MESSAGE` added to that one prompt, as the nudge is. The
      conversation carries on. The same change is blocked at most `MAX_BLOCKS_PER_CHANGE` times;
      the next time, it runs and is recorded as ``overridden``. After `MAX_BLOCKS` blocks in one
      conversation nothing more is blocked (``logged``), so a conversation always ends.
      After-the-fact findings (``live=False``) are only logged.

    Every decision is kept in ``blocks``, one entry per flagged call, so a run can count blocks,
    good changes blocked wrongly (in a conversation that then passed) and the extra agent turns
    blocks cost (``reasks``).
    """

    mode: str = "log"
    write_tools: frozenset[str] = frozenset()
    policy: str = ""
    checker: Any = None  # (conversation, call, policy) -> ActionFinding | None; default check_action
    blocks: list[dict[str, Any]] = field(default_factory=list)
    reasks: list[dict[str, Any]] = field(default_factory=list)  # each block message, as the agent saw it
    error: str | None = None  # a checker error, which switched the check off (`Watcher.before_step`)
    _counts: dict[str, int] = field(default_factory=dict)
    _message: str = BLOCK_MESSAGE  # the last block's message: its rule's own, or BLOCK_MESSAGE

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"monitor mode must be one of {MODES}, not {self.mode!r}")
        self.write_tools = frozenset(self.write_tools)
        if self.checker is None:
            from reflexive.actions import check_action

            self.checker = check_action

    def check(self, view: Sequence[Message], reply: Message, agent_turn: int) -> bool:
        """True if ``reply`` must not be sent (act mode and a live finding). Records every finding.

        ``view``: the agent's view before this reply. ``agent_turn``: this reply's number (1 = first).
        """
        if self.mode == "off":
            return False
        entries: list[tuple[str, dict[str, Any]]] = []  # (the change, its log entry)
        for k, call in enumerate(reply.get("tool_calls") or []):
            bare = _bare(call)
            if bare["name"] not in self.write_tools:
                continue
            finding = self.checker(view, call, self.policy)
            if finding is None:
                continue
            entry = {
                "agent_turn": agent_turn,
                "call": k,
                "tool": bare["name"],
                "arguments": bare["arguments"],
            }
            entry.update(finding.to_dict(), action="logged")
            key = _change_key(call)
            if self.mode == "act" and finding.live and self.n_blocked < MAX_BLOCKS:
                seen = self._counts.get(key, 0)
                entry["action"] = "blocked" if seen < MAX_BLOCKS_PER_CHANGE else "overridden"
            entries.append((key, entry))
        block = any(e["action"] == "blocked" for _, e in entries)
        if block:  # the first blocked call's rule words the re-ask (meta-healing rules have their own)
            self._message = (
                next(e.get("message") for _, e in entries if e["action"] == "blocked") or BLOCK_MESSAGE
            )
        for key, e in entries:
            if e["action"] == "blocked":
                self._counts[key] = self._counts.get(key, 0) + 1
            elif block and e["action"] == "overridden":
                e["action"] = "held"  # its reply was blocked for another call, so it didn't run either
        self.blocks += [e for _, e in entries]
        return block

    @property
    def n_blocked(self) -> int:
        return sum(e["action"] == "blocked" for e in self.blocks)

    def reasked(self, prompt: list[Message], agent_turn: int) -> list[Message]:
        """The prompt with `BLOCK_MESSAGE` as a last user turn, recorded as `Guard.nudged` records
        the nudge. The blocked reply isn't in it: τ²-bench never saw it, and an OpenAI tool call
        without its result isn't a valid message."""
        last = prompt[-1] if prompt else {}
        self.reasks.append(
            {
                "text": self._message,
                "role": "user",
                "after_view_index": len(prompt) - 1,
                "after_role": last.get("role"),
                "before_agent_turn": agent_turn,
            }
        )
        return [*prompt, {"role": "user", "content": self._message}]

    def to_dict(self) -> dict[str, Any]:
        counts = {
            a: sum(e["action"] == a for e in self.blocks) for a in ("blocked", "overridden", "held", "logged")
        }
        return {
            "mode": self.mode,
            "counts": counts,
            "blocks": self.blocks,
            "reasks": self.reasks,
            # its own copy: `Watcher.error` is shared with the monitor and the audit log
            "error": self.error,
            "switched_off_by_error": self.error is not None,
        }


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

    ``mode`` is the loop monitor's (`Guard`); ``actions`` is the action guard's (`ActionGuard`,
    SEL-89), set apart so that one can act while the other only logs (Stage 2 keeps the loop
    monitor at ``log`` for every arm).
    """

    def __init__(
        self,
        mode: str,
        tools: list[dict],
        audit_log: Path | str | None = None,
        trajectory_id: str = "",
        customer_tools: Sequence[str] = (),
        write_tools: Collection[str] = (),
        policy: str = "",
        checker: Any = None,
        actions: str = "log",
    ) -> None:
        self.guard = Guard(mode)
        self.actions = ActionGuard(actions, frozenset(write_tools), policy, checker)
        self.monitor = RuntimeMonitor(tools, customer_tools) if mode != "off" else None
        self.writer = AuditWriter(audit_log, trajectory_id) if audit_log else None
        self.error: str | None = None

    def before_step(self, view: Sequence[Message], reply: Message, agent_turn: int) -> bool:
        """True if ``reply`` is blocked (SEL-89): don't send it, re-ask with `ActionGuard.reasked`.

        A checker error is kept in ``error`` and switches the check off, as a monitor error does:
        the call then runs."""
        if self.actions.mode == "off":
            return False
        try:
            return self.actions.check(view, reply, agent_turn)
        except Exception as e:  # noqa: BLE001 - the guard must never break a conversation
            self.error = self.actions.error = f"{type(e).__name__}: {str(e)[:200]}"
            self.actions.mode = "off"
            return False

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
            "actions": self.actions.to_dict(),
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
