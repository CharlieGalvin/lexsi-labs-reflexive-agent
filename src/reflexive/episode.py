"""One τ²-bench conversation that our code drives, turn by turn (SEL-32).

    ep = Tau2Episode("telecom", task_id)          # customer: GPT-4.1 (τ²-bench's default)
    ep.reset()
    while not ep.done:
        messages = ep.prompt()                    # system prompt + conversation, fitted to 8K
        if ep.done:                               # didn't fit even after trimming → ended
            break
        reply = my_model(messages, tools=ep.tool_schemas)   # {"role": "assistant", ...}
        ep.step(reply)
    record = ep.record()                          # trajectory + outcome, ready to save as JSON

Built on τ²-bench's own Gym environment (`tau2.gym.AgentGymEnv`): τ²-bench runs the simulated
customer, the tools, the database and the customer's own tool calls exactly as in `tau2 run`;
we supply each agent turn. Two changes to the Gym environment:

- **No scoring inside the run.** The Gym environment grades every finished conversation with
  the LLM judges on, which fails for retail and airline without an OpenAI key and then loses
  the conversation. We grade afterwards with `reflexive.outcome.score` (one scorer, with a
  `judges` switch).
- **No extra `done` tool.** The Gym agent gets a `done` tool the official agent doesn't have.
  `tool_schemas` hides it, so the model sees exactly the official agent's tools. We still use
  it internally to end a conversation that no longer fits the context window.

**Agent turns** are OpenAI-style dicts: ``{"role": "assistant", "content": "..."}`` for a
message to the customer, or ``{"role": "assistant", "tool_calls": [...]}`` for one or more
tool calls. τ²-bench's instructions say "send a message *or* make a tool call, not both";
a turn with both is passed through as it is, as in an official run.

**Tool results** go back to the model as compact JSON (no indentation): it's the same data in
fewer tokens, which matters with Tiny Aya's 8,192-token window.

**Context guard.** `prompt()` keeps the prompt (conversation plus tool definitions) within
``max_context - reserve`` tokens, where ``reserve`` is room left for the model's reply, by
replacing the oldest tool results with a short placeholder (in what the model sees only: the
saved trajectory keeps them in full). If it still doesn't fit, the conversation ends with the
termination reason ``context_window_exceeded``: it scores 0 and is flagged ``context_overflow``,
so it is reported separately and never treated as a hallucination to heal. Token counting is
pluggable (``count_tokens``); the default is a rough estimate (characters / 4), to be replaced
with Tiny Aya's tokenizer once everyone has access.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable
from typing import Any

from reflexive.outcome import from_openai, scorer, to_openai

TRUNCATED = "[older tool result removed to fit the context window]"
DEFAULT_MAX_STEPS = 200  # same as `tau2 run`
DEFAULT_MAX_ERRORS = 10


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token count: characters / 4, plus a little per message. Replace with a tokenizer."""
    chars = 0
    for m in messages:
        chars += len(m.get("content") or "") + 16
        for tc in m.get("tool_calls") or []:
            chars += len(json.dumps(tc))
    return chars // 4


def compact_json(text: str | None) -> str | None:
    """Re-serialise a JSON tool result without whitespace; leave non-JSON text unchanged."""
    if not text:
        return text
    try:
        return json.dumps(json.loads(text), separators=(",", ":"), ensure_ascii=False)
    except (ValueError, TypeError):
        return text


def fit_context(messages: list[dict], budget: int,
                count_tokens: Callable[[list[dict]], int] = estimate_tokens
                ) -> tuple[list[dict], int, bool]:
    """Trim the oldest tool results until ``messages`` fits in ``budget`` tokens.

    Returns ``(messages, n_trimmed, fits)``. The system prompt, every agent and customer turn,
    and the most recent tool result are never touched.
    """
    out = [dict(m) for m in messages]
    tool_idx = [i for i, m in enumerate(out) if m["role"] == "tool"]
    trimmed = 0
    for i in tool_idx[:-1]:  # oldest first, keep the latest result
        if count_tokens(out) <= budget:
            break
        if out[i].get("content") != TRUNCATED:
            out[i]["content"] = TRUNCATED
            trimmed += 1
    return out, trimmed, count_tokens(out) <= budget


class Tau2Episode:
    """One τ²-bench conversation: our code plays the agent, τ²-bench plays everything else."""

    def __init__(self, domain: str, task_id: str, *, user_llm: str | None = None,
                 user_llm_args: dict | None = None, max_steps: int = DEFAULT_MAX_STEPS,
                 max_context: int = 8192, reserve: int = 1024,
                 count_tokens: Callable[[list[dict]], int] = estimate_tokens,
                 language: str = "en", seed: int | None = None):
        from reflexive.llm import default_model, register_openrouter_prices

        register_openrouter_prices()
        self.domain, self.task_id, self.language, self.seed = domain, task_id, language, seed
        self.user_llm = user_llm or default_model()
        self.user_llm_args = user_llm_args if user_llm_args is not None else {"temperature": 0.0}
        self.max_steps, self.max_context, self.reserve = max_steps, max_context, reserve
        self.count_tokens = count_tokens
        scorer(domain, language).task(task_id)  # fail fast on an unknown task id
        self._env = None
        self._view: list[Any] = []  # what the agent has seen so far (τ²-bench messages)
        self.flags: dict[str, Any] = {"context_overflow": False, "tool_results_trimmed": 0,
                                      "peak_prompt_tokens": 0}
        self.done = False
        self._overflowed = False
        self.started = self.ended = None

    # --- Lifecycle -----------------------------------------------------------------------

    def reset(self) -> list[dict]:
        """Start the conversation. Returns the new messages (normally the customer's opener)."""
        env = _NoScoreAgentGymEnv(
            domain=self.domain, task_id=self.task_id, max_steps=self.max_steps,
            user_llm=self.user_llm, user_llm_args=self.user_llm_args, language=self.language,
        )
        self._env = env
        self.started = time.time()
        env.reset(seed=self.seed)
        tools = [t for t in env._get_tools() if t.name != "done"]
        self._tools = tools
        self.policy = env._agent.domain_policy  # includes τ²-bench's language guidance
        return self._sync()

    def step(self, reply: dict) -> list[dict]:
        """Send one agent turn. Returns what happened next: tool results and/or customer replies."""
        if self.done:
            raise RuntimeError("The conversation has ended")
        if reply.get("role", "assistant") != "assistant":
            raise ValueError("An agent turn must have role 'assistant'")
        if not reply.get("tool_calls") and not (reply.get("content") or "").strip():
            # τ²-bench rejects this and the whole run crashes, losing the conversation
            raise ValueError("An agent turn needs a message or at least one tool call (got neither)")
        (msg,) = from_openai([{**reply, "role": "assistant"}])
        self._env._agent.set_action(msg)
        self._wait_for_turn()
        new = self._sync()
        if new and new[0]["role"] == "assistant":  # our own turn, echoed back
            new = new[1:]
        return new

    def _wait_for_turn(self) -> None:
        env = self._env
        while not env._simulation_done.is_set() and not env._agent.is_agent_turn:
            env._simulation_done.wait(timeout=0.01)

    def _sync(self) -> list[dict]:
        """New agent-visible messages since the last call, as OpenAI-style dicts."""
        if self._env._simulation_done.is_set():
            self.done = True
            self.ended = time.time()
        obs = self._env._agent.observation
        new = obs[len(self._view):]
        self._view = list(obs)
        return [self._for_model(m) for m in to_openai(new)]

    @staticmethod
    def _for_model(m: dict) -> dict:
        if m["role"] == "tool":
            return {**m, "content": compact_json(m["content"])}
        return m

    # --- What the model sees ------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """Exactly τ²-bench's agent prompt: its instructions + this domain's policy."""
        from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

        return SYSTEM_PROMPT.format(agent_instruction=AGENT_INSTRUCTION,
                                    domain_policy=self.policy)

    @property
    def tool_schemas(self) -> list[dict]:
        """The official agent's tools, as OpenAI function schemas (no `done` tool)."""
        return [t.openai_schema for t in self._tools]

    def messages(self) -> list[dict]:
        """System prompt + the whole conversation so far as the agent sees it (untrimmed)."""
        convo = [self._for_model(m) for m in to_openai(self._view)]
        return [{"role": "system", "content": self.system_prompt}, *convo]

    def tool_tokens(self) -> int:
        """Tokens the tool definitions take up in the prompt (counted with ``count_tokens``)."""
        return self.count_tokens([{"role": "system", "content": json.dumps(self.tool_schemas)}])

    def prompt(self) -> list[dict]:
        """`messages()` fitted to ``max_context - reserve`` tokens, tool definitions included.

        If it can't fit, the conversation is ended (``context_window_exceeded``) and
        ``done`` becomes True; the returned messages are then the trimmed attempt.
        """
        if self.done:
            raise RuntimeError("The conversation has ended")
        budget = self.max_context - self.reserve - self.tool_tokens()
        full = self.messages()
        msgs, trimmed, fits = fit_context(full, budget, self.count_tokens)
        self.flags["tool_results_trimmed"] = max(self.flags["tool_results_trimmed"], trimmed)
        self.flags["peak_prompt_tokens"] = max(  # untrimmed size: evidence for the telecom decision
            self.flags["peak_prompt_tokens"], self.count_tokens(full) + self.tool_tokens())
        if not fits:
            self._end_for_context()
        return msgs

    def _end_for_context(self) -> None:
        self._overflowed = True
        self.flags["context_overflow"] = True
        self._send_done()

    def finish(self) -> None:
        """End the conversation from the agent's side (termination ``agent_stop``).

        For test drivers. A model ends a conversation the way τ²-bench's agent does: by
        writing ``###STOP###`` in a message, or by transferring to a human agent.
        """
        if not self.done:
            self._send_done()

    def _send_done(self) -> None:
        from tau2.data_model.message import AssistantMessage, ToolCall

        stop = AssistantMessage(role="assistant", content=None, tool_calls=[
            ToolCall(id="end", name="done", arguments={})])
        self._env._agent.set_action(stop)
        self._wait_for_turn()
        self._sync()

    # --- Tools as Python functions (AgentTune `tools=`) -----------------------------------

    def tool_functions(self) -> list[Callable[..., str]]:
        """Each official tool as a Python function that runs a tool-call turn in this conversation.

        Same name, arguments and description as the τ²-bench tool. Calling one sends a single
        tool call as the agent's turn and returns the (compact JSON) result. Only tool turns
        are possible this way; messages to the customer still go through `step`.
        """
        return [self._tool_function(t) for t in self._tools]

    def _tool_function(self, tool) -> Callable[..., str]:
        episode = self

        def call(**kwargs: Any) -> str:
            reply = {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"call_{len(episode._view)}", "name": tool.name, "arguments": kwargs}]}
            new = episode.step(reply)
            results = [m["content"] for m in new if m["role"] == "tool"]
            return results[0] if results else ""

        call.__name__ = tool.name
        call.__qualname__ = tool.name
        call.__doc__ = tool._get_description()
        sig = tool.__signature__
        call.__signature__ = sig.replace(parameters=[
            p for p in sig.parameters.values() if p.kind != inspect.Parameter.VAR_KEYWORD])
        return call

    # --- Result --------------------------------------------------------------------------

    @property
    def termination(self) -> str:
        if self._overflowed:
            return "context_window_exceeded"
        run = self._env._simulation_run if self._env else None
        if run is None:
            return "unexpected_error"
        return run.termination_reason.value

    def trajectory(self) -> list[dict]:
        """The full conversation, including the customer's own tool calls, as OpenAI-style dicts.

        If τ²-bench's run crashed, this is the conversation up to the crash.
        """
        run = self._env._simulation_run
        if run is not None and run.messages:
            return to_openai(run.messages)
        try:  # crashed run: recover what the orchestrator had so far
            return to_openai(self._env._orchestrator.get_messages())
        except Exception:  # noqa: BLE001 - best effort; an empty trajectory is still valid
            return []

    def record(self, judges: bool = True, **extra: Any) -> dict:
        """Everything about this conversation, ready to save as one JSON line.

        Provisional format until the data contract (SEL-35) is agreed.
        """
        if not self.done:
            raise RuntimeError("The conversation hasn't ended yet")
        traj = self.trajectory()
        outcome = scorer(self.domain, self.language).score(
            self.task_id, traj, self.termination, judges=judges)
        run = self._env._simulation_run
        return {
            "format": "reagent.tau2.episode/v0",
            "domain": self.domain, "task_id": self.task_id, "language": self.language,
            "termination": self.termination,
            "outcome": {"passed": outcome.passed, "reward": outcome.reward,
                        "checks": outcome.checks, "judged": outcome.judged,
                        "skipped": outcome.skipped, "note": outcome.note},
            "flags": dict(self.flags),
            "user_llm": self.user_llm, "seed": self.seed,
            "max_context": self.max_context, "reserve": self.reserve,
            "user_cost": getattr(run, "user_cost", None) if run else None,
            "duration_s": round((self.ended or time.time()) - (self.started or time.time()), 2),
            "messages": traj,
            **extra,
        }


def _no_score_env_class():
    from tau2.gym.gym_agent import AgentGymEnv

    class NoScoreAgentGymEnv(AgentGymEnv):
        """τ²-bench's Gym environment, minus the in-run scoring (see module docstring)."""

        def _run_orchestrator(self):
            simulation_run = None
            try:
                if self._orchestrator:
                    simulation_run = self._orchestrator.run()
            except Exception as e:  # same handling as the parent: never hang the caller
                self._log(f"Orchestrator error: {e}", "ERROR")
            finally:
                self._simulation_run = simulation_run
                self._simulation_done.set()

        def _get_reward(self):
            return 0.0, "{}"

    return NoScoreAgentGymEnv


_REGISTERED = False


def _NoScoreAgentGymEnv(**kwargs):  # noqa: N802 - lazy import keeps tau2 optional
    global _REGISTERED
    if not _REGISTERED:  # registers τ²-bench's Gym agent/customer roles, once per process
        from tau2.gym import register_gym_agent

        register_gym_agent()
        _REGISTERED = True
    return _no_score_env_class()(**kwargs)
