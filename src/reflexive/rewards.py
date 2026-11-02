"""Reward functions for GRPO on tool-calling trajectories.

Signature follows TRL / AgentTune: ``fn(prompts, completions, **kwargs) -> list[float]``.
Dataset columns (e.g. ``answer``) arrive as kwargs, one value per completion.

Design notes — read before changing weights
-------------------------------------------
* Several AgentTune built-ins (``search_grounding_reward``, ``exploration_reward``)
  pay MORE for more tool calls. That is the opposite of our thesis, so we don't use them.
* An efficiency reward alone is easy to game: answer without calling any tool. For a
  yes/no task that is a coin flip that still scores. So ``efficiency_reward`` pays only
  when the answer is correct AND grounded in at least one successful tool result.
* ``groundedness_reward`` is the hallucination term: confident claims with no tool
  evidence are penalised; abstaining on genuinely unanswerable questions is rewarded;
  abstaining on answerable ones costs a little (so the model doesn't learn to refuse).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from . import trajectory as T

RewardFn = Callable[..., list[float]]


def _gold(answer: Sequence[Any] | None, i: int) -> str | None:
    if answer is None:
        return None
    return None if answer[i] is None else str(answer[i])


def correctness_reward(prompts, completions, answer=None, **kwargs) -> list[float]:
    """1.0 if the final answer matches the gold answer (or correctly abstains), else 0."""
    out = []
    for i, c in enumerate(completions):
        traj = T.parse(c)
        out.append(1.0 if T.is_correct(traj.final_answer, _gold(answer, i)) else 0.0)
    return out


def make_efficiency_reward(step_budget: int = 3, min_calls: int = 1,
                           dup_penalty: float = 0.3, error_penalty: float = 0.2,
                           over_budget_penalty: float = 0.15) -> RewardFn:
    """Pays for reaching a correct, grounded answer with few, non-repeated, non-failing calls."""

    def efficiency_reward(prompts, completions, answer=None, **kwargs) -> list[float]:
        out = []
        for i, c in enumerate(completions):
            traj = T.parse(c)
            gold = _gold(answer, i)
            unanswerable = gold is not None and gold.upper() == T.UNANSWERABLE
            correct = T.is_correct(traj.final_answer, gold)
            grounded = bool(traj.successful_results) or unanswerable
            if not correct or not grounded or traj.n_calls < (0 if unanswerable else min_calls):
                out.append(0.0)  # gate: no efficiency credit for wrong or ungrounded answers
                continue
            score = 1.0
            score -= dup_penalty * traj.n_redundant
            score -= error_penalty * traj.n_errors
            score -= over_budget_penalty * max(0, traj.n_calls - step_budget)
            out.append(max(-1.0, min(1.0, score)))
        return out

    efficiency_reward.__name__ = "efficiency_reward"
    return efficiency_reward


def groundedness_reward(prompts, completions, answer=None, **kwargs) -> list[float]:
    """Hallucination term. Range roughly [-1, 1]."""
    out = []
    for i, c in enumerate(completions):
        traj = T.parse(c)
        gold = _gold(answer, i)
        unanswerable = gold is not None and gold.upper() == T.UNANSWERABLE
        if not traj.final_answer:
            out.append(-0.5)  # ran out of steps / never answered
        elif traj.abstained:
            out.append(1.0 if unanswerable else -0.25)
        elif unanswerable:
            out.append(-1.0)  # confident answer to an unanswerable question
        elif not traj.successful_results:
            out.append(-1.0)  # claim with zero tool evidence
        else:
            bad = T.unsupported_numbers(traj.final_answer, traj.successful_results)
            out.append(max(-1.0, 0.5 - 0.5 * len(bad)))
    return out


def make_validity_reward(tool_names: Sequence[str]) -> RewardFn:
    """Fraction of calls that name a real tool with well-formed JSON arguments."""
    names = set(tool_names)

    def tool_validity_reward(prompts, completions, **kwargs) -> list[float]:
        out = []
        for c in completions:
            traj = T.parse(c)
            if not traj.calls:
                out.append(0.0)
                continue
            ok = sum(1 for call in traj.calls if call.well_formed and call.name in names)
            out.append(ok / traj.n_calls)
        return out

    tool_validity_reward.__name__ = "tool_validity_reward"
    return tool_validity_reward


def build_reward_funcs(cfg: dict[str, Any], tool_names: Sequence[str]) -> tuple[list[RewardFn], list[float]]:
    """Build the reward list + weights from the ``rewards:`` block of a config."""
    registry: dict[str, RewardFn] = {
        "correctness": correctness_reward,
        "efficiency": make_efficiency_reward(**cfg.get("efficiency_kwargs", {})),
        "groundedness": groundedness_reward,
        "tool_validity": make_validity_reward(tool_names),
    }
    weights = cfg.get("weights", {"correctness": 1.0, "efficiency": 0.5,
                                  "groundedness": 0.5, "tool_validity": 0.2})
    unknown = set(weights) - set(registry)
    if unknown:
        raise ValueError(f"unknown reward(s) {unknown}; known: {sorted(registry)}")
    names = [n for n, w in weights.items() if w]
    return [registry[n] for n in names], [float(weights[n]) for n in names]
