"""Model names and prices for the API models we use alongside τ²-bench (customer, test agent)."""

from __future__ import annotations

import os

OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]


def register_openrouter_prices() -> None:
    """Copy OpenAI list prices onto OpenRouter's names, so LiteLLM can cost OpenRouter calls.

    Without this, τ²-bench logs "This model isn't mapped yet" on every message and records $0.
    OpenRouter passes OpenAI's prices through; treat the result as an estimate.
    """
    import litellm

    extra = {}
    for model in OPENAI_MODELS:
        base = litellm.model_cost.get(model)
        if not base:
            continue
        entry = {**base, "litellm_provider": "openrouter"}
        for alias in (f"openrouter/openai/{model}", f"openrouter/{model}"):
            if alias not in litellm.model_cost:
                extra[alias] = entry
    if extra:
        litellm.register_model(extra)


def default_model(model: str = "gpt-4.1") -> str:
    """`gpt-4.1` via OpenAI if OPENAI_API_KEY is set, else via OpenRouter (pinned snapshot on OpenAI)."""
    if os.environ.get("OPENAI_API_KEY"):
        return "gpt-4.1-2025-04-14" if model == "gpt-4.1" else model
    if os.environ.get("OPENROUTER_API_KEY"):
        return f"openrouter/openai/{model}"
    raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY (see .env.example)")
