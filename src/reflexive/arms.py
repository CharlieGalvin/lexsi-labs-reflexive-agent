"""The experiment arms and one rollout of an arm on one τ²-bench task (SEL-41).

    arm = get_arm("C")                         # from configs/arms.yaml
    agent = make_student(arm.student)          # the student as a `Tau2Episode` agent
    rec = run_episode(arm, agent, "retail", task_id)

Every arm runs the same loop with the same student, customer and context settings. What an
arm changes is in its config entry: for now only ``prompt_variant`` (Arm A: τ²-bench's prompt;
Arm C: plus `reflexive.episode.CAREFUL_INSTRUCTION`). The record is `Tau2Episode.record()`
plus the arm, the student and per-conversation diagnostics, so the same scoring applies to
every arm.

**Data contract (SEL-35, agreed 4 Nov).** The student works in Cohere's ``Action:`` format, with
the same prompt Aya's baseline used (`TeacherClient`, ``prompted`` mode, ``cohere`` style), for
evaluation and training alike. Each record also keeps ``agent_view``: the conversation exactly
as the agent saw it, system prompt included (the contract's input). It's saved OpenAI-style,
like the teacher runs (SEL-58), and `reflexive.teacher.to_prompted` renders it as the
``Action:`` text the model read.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reflexive.episode import Tau2Episode
from reflexive.teacher import (
    ContextTooLong,
    RateLimiter,
    TeacherClient,
    tool_errors,
    unseen_arguments,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "arms.yaml"
STUDENT_KEYS = {
    "name",
    "model",
    "api_base",
    "api_key_env",
    "mode",
    "style",
    "temperature",
    "max_tokens",
    "max_context",
    "reserve",
    "rpm",
}


@dataclass(frozen=True)
class Arm:
    key: str  # "A", "C", ...
    name: str
    prompt_variant: str
    student: dict
    customer: str | None = None
    description: str = ""


def load_config(path: str | Path | None = None) -> dict:
    import yaml

    return yaml.safe_load(Path(path or CONFIG_PATH).read_text())


def get_arm(key: str, path: str | Path | None = None) -> Arm:
    """One arm from the config, with the shared student and customer settings."""
    cfg = load_config(path)
    arms = cfg.get("arms") or {}
    if key not in arms:
        raise KeyError(f"Unknown arm {key!r}; the config has {', '.join(arms) or 'none'}")
    unknown = set(cfg["student"]) - STUDENT_KEYS
    if unknown:
        raise ValueError(f"Unknown student settings: {', '.join(sorted(unknown))}")
    a = arms[key]
    return Arm(
        key=key,
        name=a["name"],
        prompt_variant=a.get("prompt_variant", "default"),
        student=dict(cfg["student"]),
        customer=cfg.get("customer"),
        description=a.get("description", ""),
    )


def make_student(student: dict, *, limiter: RateLimiter | None = None, **kwargs: Any) -> TeacherClient:
    """The student model as an agent (`turn(messages, tool_schemas)`). Same client as the teacher.

    ``limiter``: share one `RateLimiter` between conversations run at once, so the student's
    ``rpm`` cap holds for the whole run (each client otherwise spaces only its own calls).
    """
    key_env = student.get("api_key_env")
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key and "completion" not in kwargs:
        raise RuntimeError(f"Set {key_env} in .env for the student ({student['model']})")
    client = TeacherClient(
        student.get("mode", "prompted"),
        style=student.get("style", "cohere"),
        model=student["model"],
        api_base=student["api_base"],
        api_key=api_key or "none",
        temperature=student.get("temperature", 0.0),
        max_tokens=student.get("max_tokens", 1024),
        rpm=student.get("rpm") or 0,  # not TEACHER_RPM: the teacher's cap isn't the student's
        prices=(0.0, 0.0) if not key_env else None,
        **kwargs,
    )
    if limiter is not None:
        client.limiter = limiter
    return client


def make_episode(arm: Arm, domain: str, task_id: str, **kwargs: Any) -> Tau2Episode:
    """The arm's `Tau2Episode`: its prompt variant, the student's window, never hints."""
    s = arm.student
    return Tau2Episode(
        domain,
        task_id,
        user_llm=kwargs.pop("user_llm", arm.customer),
        max_context=s.get("max_context", 128_000),
        reserve=s.get("reserve", 4_096),
        prompt_variant=arm.prompt_variant,
        hints="none",
        **kwargs,
    )


def run_episode(
    arm: Arm,
    agent: Any,
    domain: str,
    task_id: str,
    *,
    judges: bool = True,
    sample: int = 0,
    monitor: str = "log",
    audit_log: Path | str | None = None,
    **episode_kwargs: Any,
) -> dict:
    """One conversation of ``arm`` on one task. Never raises for a failed conversation.

    A crash mid-conversation is recorded (``error``) and scored on what happened so far; a crash
    before it started is a failed attempt with ``termination="unexpected_error"``.

    After every turn the runtime monitor (`reflexive.monitor`, SEL-62) checks the agent's view.
    ``monitor``: ``log`` (default; the arms only record what it finds, in ``rec["monitor"]``),
    ``act`` (a ``no_progress`` loop gets a nudge, then a stop: an ablation of its own) or ``off``.
    ``audit_log``: a JSONL file the conversation is appended to after every turn, in the
    audit-log format AgentTune's ``FullClosedLoop`` reads.
    """
    from reflexive.monitor import Guard, Watcher

    Guard(monitor)  # refuse an unknown mode before the conversation starts
    ep = make_episode(arm, domain, task_id, **episode_kwargs)
    watch: Watcher | None = None
    start = time.monotonic()
    error = None
    try:
        ep.reset()
        # the live request's tools: the monitor knows no dataset rules. The customer's tool
        # names, when the domain has them, only label a call (customer_tool, as classify_record).
        watch = Watcher(
            monitor,
            ep.tool_schemas,
            audit_log,
            f"{arm.key}/{domain}/{task_id}/{sample}",
            customer_tools=_customer_tools(domain),
        )
        nudge = False
        while not ep.done:
            messages = ep.prompt()
            if ep.done:
                break
            if nudge:
                messages, nudge = watch.guard.nudged(messages), False
            try:
                ep.step(agent.turn(messages, ep.tool_schemas))
            except ContextTooLong:  # the API's own token count was over the window
                ep.end_for_context()
            act = watch.after_turn(ep.messages())
            nudge = act == "nudge"
            if act == "stop":  # the guard's stop, recorded as termination monitor_stop
                ep.stop_for_monitor()
    except Exception as e:  # noqa: BLE001 - one bad conversation shouldn't stop the run
        error = f"{type(e).__name__}: {str(e)[:300]}"
        if ep.has_env and not ep.done:
            ep.finish()
    if watch is not None and ep.has_env:
        watch.close(ep.messages())
    usage = agent.usage.as_dict() if hasattr(agent, "usage") else None
    extra = {
        "arm": arm.key,
        "arm_name": arm.name,
        "student": arm.student.get("name"),
        "agent_llm": arm.student.get("model"),
        "agent_api_base": arm.student.get("api_base"),
        "agent_mode": arm.student.get("mode", "prompted"),
        "agent_style": arm.student.get("style", "cohere"),
        "agent_usage": usage,
        "sample": sample,
        "error": error,
        "monitor": watch.to_dict() if watch is not None else {"mode": monitor, "alarms": [], "error": None},
    }
    if ep.done:
        rec = ep.record(judges=judges, **extra)
    else:
        rec = {
            "format": "reagent.tau2.episode/v0",
            "domain": domain,
            "task_id": task_id,
            "prompt_variant": arm.prompt_variant,
            "termination": "unexpected_error",
            "outcome": {
                "passed": False,
                "reward": 0.0,
                "checks": {},
                "judged": False,
                "skipped": [],
                "note": error,
            },
            "flags": dict(ep.flags),
            "messages": [],
            **extra,
        }
    msgs = rec["messages"]
    rec["diagnostics"] = {
        "tools_ran": sum(len(m.get("tool_calls") or []) for m in msgs if m["role"] == "assistant"),
        "unseen_arguments": unseen_arguments(msgs),
        **tool_errors(msgs),
    }
    if ep.has_env:  # what the agent was actually shown (system prompt included)
        try:
            rec["agent_view"] = ep.messages()
        except Exception:  # noqa: BLE001 - never lose a conversation over this extra
            pass
    raw = getattr(agent, "raw_turns", None)
    if raw:  # prompted models (Aya): the raw replies, to review how calls were written and read
        rec["raw_turns"] = raw
    if watch is not None:  # the outcome line: the detector learns only from failed conversations
        outcome = rec.get("outcome")
        watch.finish(isinstance(outcome, dict) and bool(outcome.get("passed")), str(rec.get("termination")))
    rec["wall_s"] = round(time.monotonic() - start, 1)
    return rec


def _customer_tools(domain: str) -> list[str]:
    """The domain's customer-only tool names, or none if it isn't a known τ²-bench domain."""
    from reflexive.hallucination import domain_tools

    try:
        return list(domain_tools(domain)[1])
    except Exception:  # noqa: BLE001 - only a label: never stop a conversation over it
        return []
