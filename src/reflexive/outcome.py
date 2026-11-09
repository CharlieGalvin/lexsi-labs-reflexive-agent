"""Outcome checker: pass/fail for any τ²-bench trajectory, using τ²-bench's own scoring (SEL-39).

    from reflexive.outcome import score
    result = score("telecom", task_id, messages, termination="user_stop")
    result.passed, result.reward, result.checks

Nothing here re-implements grading. The trajectory is wrapped as a τ²-bench ``SimulationRun``
and passed to ``tau2.evaluator.evaluate_simulation`` with the same task definitions,
environment (language-pack database and policy) and judge settings that ``tau2 run`` uses.

**Trajectory format.** ``messages`` can be τ²-bench ``Message`` objects (or their dicts), or
OpenAI-style chat dicts, the format our harness writes:

- ``{"role": "assistant", "content": ..., "tool_calls": [...]}`` is the agent.
- ``{"role": "user", ...}`` is the simulated customer. In telecom the customer has tools
  too (toggling airplane mode on their phone), so a ``user`` message may also carry
  ``tool_calls``. Keep these: τ²-bench replays them to rebuild the end state.
- ``{"role": "tool", "tool_call_id": ..., "content": ...}`` is a tool result. It must directly
  follow the message whose call it answers, as in the OpenAI format.
- ``system`` messages are ignored (the policy prompt is not part of what is graded).

``tool_calls`` entries may be OpenAI-shaped (``{"id", "function": {"name", "arguments": "<json>"}}``)
or flat (``{"id", "name", "arguments": {...}}``).

**Passing.** ``passed`` means reward == 1.0, τ²-bench's own rule (pass^1). The reward is the
product of the checks in the task's ``reward_basis``. A conversation that did not end with
the agent or the customer stopping (max steps, too many errors, our context guard...) scores 0.

**Judges.** Retail's NL assertions and airline's communicate check are graded by LLM judges
pinned to OpenAI models (need ``OPENAI_API_KEY``, cost money, can vary between calls).
``judges=False`` scores only the deterministic checks (DB, ENV_ASSERTION, ACTION) and lists
what was skipped in ``skipped``. Telecom never uses a judge.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Literal, cast

JUDGED = {"NL_ASSERTION", "COMMUNICATE"}
Requestor = Literal["user", "assistant"]  # who made a tool call (τ²-bench ToolMessage)
NORMAL_STOPS = {"agent_stop", "user_stop"}
# Our own endings, which τ²-bench doesn't have: each is scored as the τ²-bench ending it sends.
# ``monitor_stop``: the runtime monitor's guard ended a no_progress loop (SEL-69). The harness
# sends τ²-bench the agent's stop, so it is graded as if the agent had ended the conversation,
# but the record keeps its own ending so those runs can be reported on their own.
MONITOR_STOP = "monitor_stop"
SCORED_AS = {MONITOR_STOP: "agent_stop"}


@dataclass
class Outcome:
    """Result of scoring one trajectory."""

    passed: bool
    reward: float
    checks: dict[str, float]  # reward breakdown by check, e.g. {"DB": 1.0, "NL_ASSERTION": 0.0}
    judged: bool  # an LLM judge contributed to `reward`
    skipped: list[str] = field(default_factory=list)  # judged checks left out (judges=False)
    note: str | None = None  # e.g. why a premature stop scored 0
    reward_info: Any = None  # τ²-bench's full RewardInfo, for per-assertion detail

    @property
    def verified(self) -> bool:
        """The verdict holds whatever the skipped judged checks would say: nothing was skipped,
        or it's a failure (a check that ran failed, or an early ending, which scores 0).

        A pass with judged checks skipped is a best case, never a verified pass. That covers a
        task whose basis is judged checks only: τ²-bench then runs nothing and returns 1.0."""
        return not self.skipped or not self.passed


# --- Trajectory format -----------------------------------------------------------------------


def _tool_call(tc: dict, requestor: str) -> dict:
    if "function" in tc:  # OpenAI shape
        fn = tc["function"]
        args = fn.get("arguments") or {}
        name = fn["name"]
    else:
        args = tc.get("arguments") or {}
        name = tc["name"]
    if isinstance(args, str):
        args = json.loads(args) if args.strip() else {}
    return {
        "id": tc.get("id") or "",
        "name": name,
        "arguments": args,
        "requestor": tc.get("requestor", requestor),
    }


def from_openai(messages: list[Any]) -> list[Any]:
    """Convert OpenAI-style dicts (or τ²-bench message dicts/objects) to τ²-bench messages."""
    from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage

    out: list[Any] = []
    requestor_of: dict[str, str] = {}  # tool_call id -> who made the call
    for m in messages:
        if not isinstance(m, dict):  # already a τ²-bench Message
            out.append(m)
            for tc in getattr(m, "tool_calls", None) or []:
                requestor_of[tc.id] = tc.requestor
            continue
        role = m.get("role")
        if role == "system":
            continue
        if role in ("assistant", "user"):
            calls = [_tool_call(tc, role) for tc in (m.get("tool_calls") or [])]
            for tc in calls:
                requestor_of[tc["id"]] = tc["requestor"]
            cls = AssistantMessage if role == "assistant" else UserMessage
            out.append(cls(role=role, content=m.get("content"), tool_calls=calls or None))
        elif role == "tool":
            call_id = m.get("tool_call_id") or m.get("id") or ""
            content = m.get("content")
            if not isinstance(content, str) and content is not None:
                content = json.dumps(content)
            out.append(
                ToolMessage(
                    id=call_id,
                    role="tool",
                    content=content,
                    requestor=cast(Requestor, m.get("requestor") or requestor_of.get(call_id, "assistant")),
                    error=bool(m.get("error", False)),
                )
            )
        else:
            raise ValueError(f"Unknown message role {role!r}")
    return out


def to_openai(messages: list[Any]) -> list[dict]:
    """Convert τ²-bench messages to OpenAI-style dicts (customer tool calls kept on `user`)."""
    out = []
    for m in messages:
        if getattr(m, "tool_messages", None) is not None:  # MultiToolMessage: several results
            out += to_openai(m.tool_messages)
            continue
        if m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.id, "content": m.content})
            continue
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        out.append(d)
    return out


# --- Scoring ---------------------------------------------------------------------------------


class Tau2Scorer:
    """Task definitions + evaluation environment for one domain, set up as `tau2 run` does."""

    def __init__(self, domain: str, language: str = "en") -> None:
        from tau2.data_model.simulation import TextRunConfig
        from tau2.multilingual.runtime import language_environment_kwargs, load_run_language

        # Same config path as `tau2 run --domain <domain>`; model names are unused here.
        self.config = TextRunConfig(domain=domain, language=language, llm_agent="-", llm_user="-")
        loaded = load_run_language(self.config)
        self.domain = domain
        self.tasks = loaded.task_index
        self.env_kwargs = language_environment_kwargs(loaded) or None
        self.communicate_scoring = self.config.communicate_scoring
        self.customer_language = None  # English runs are not cross-lingual

    def task(self, task_id: str) -> Any:  # a τ²-bench Task
        try:
            return self.tasks[task_id]
        except KeyError:
            raise KeyError(f"No {self.domain} task {task_id!r}") from None

    def score(self, task_id: str, messages: list[Any], termination: str, judges: bool = True) -> Outcome:
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        termination = SCORED_AS.get(termination, termination)
        task = self.task(task_id)
        basis = [b.value for b in (task.evaluation_criteria.reward_basis if task.evaluation_criteria else [])]
        skipped = [] if judges else [b for b in basis if b in JUDGED]
        if skipped:  # grade only the deterministic part of the basis
            crit = task.evaluation_criteria.model_copy(
                update={
                    "reward_basis": [
                        b for b in task.evaluation_criteria.reward_basis if b.value not in JUDGED
                    ]
                }
            )
            task = task.model_copy(update={"evaluation_criteria": crit})

        sim = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=task_id,
            start_time="",
            end_time="",
            duration=0.0,
            termination_reason=TerminationReason(termination),
            messages=from_openai(messages),
        )
        info = evaluate_simulation(
            simulation=sim,
            task=task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            # τ²-bench hints env_kwargs as `dict = None` and turns None into {} itself
            domain=self.domain,
            env_kwargs=self.env_kwargs,  # ty: ignore[invalid-argument-type]
            strict_replay=False,
            communicate_scoring=self.communicate_scoring,
            customer_language=self.customer_language,
        )
        checks = {getattr(k, "value", str(k)): v for k, v in (info.reward_breakdown or {}).items()}
        note = (info.info or {}).get("note") if isinstance(info.info, dict) else None
        return Outcome(
            passed=info.reward == 1.0,
            reward=info.reward,
            checks=checks,
            judged=termination in NORMAL_STOPS and any(b in JUDGED for b in basis) and not skipped,
            skipped=skipped,
            note=note,
            reward_info=info,
        )

    def gold_trajectory(self, task_id: str, drop: int | None = None) -> list[dict]:
        """The task's correct actions as an OpenAI-style conversation, with real tool outputs.

        Each expected action becomes a call by its requestor (agent or customer), followed by
        the output the environment really returns. ``drop`` leaves out the action at that
        index, to check that the scorer notices a missing step.
        """
        from tau2.data_model.message import ToolCall
        from tau2.registry import registry

        task = self.task(task_id)
        env = registry.get_env_constructor(self.domain)(**(self.env_kwargs or {}))
        ist = task.initial_state
        env.set_state(
            ist.initialization_data if ist else None, ist.initialization_actions if ist else None, []
        )
        msgs: list[dict] = []
        for i, a in enumerate(task.evaluation_criteria.actions or []):
            if i == drop:
                continue
            tc = {"id": f"call_{i}", "name": a.name, "arguments": a.arguments, "requestor": a.requestor}
            result = env.get_response(ToolCall(**tc))
            msgs.append(
                {
                    "role": "assistant" if a.requestor == "assistant" else "user",
                    "content": None,
                    "tool_calls": [tc],
                }
            )
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": result.id,
                    "content": result.content,
                    "requestor": a.requestor,
                }
            )
        msgs.append({"role": "user", "content": "Thanks, that's everything. ###STOP###"})
        return msgs

    def write_action_indices(self, task_id: str) -> list[int]:
        """Indices of the task's expected actions that change state (not reads)."""
        from tau2.registry import registry

        env = registry.get_env_constructor(self.domain)(**(self.env_kwargs or {}))
        acts = self.task(task_id).evaluation_criteria.actions or []
        return [i for i, a in enumerate(acts) if env._is_mutating_tool(a.name)]


@cache
def scorer(domain: str, language: str = "en") -> Tau2Scorer:
    """Cached per domain: loading a domain's tasks and database takes a few seconds."""
    return Tau2Scorer(domain, language)


def score(
    domain: str,
    task_id: str,
    messages: list[Any],
    termination: str,
    judges: bool = True,
    language: str = "en",
) -> Outcome:
    """Pass/fail for one trajectory. See the module docstring for the message format."""
    return scorer(domain, language).score(task_id, messages, termination, judges=judges)
