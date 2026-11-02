"""Run the τ²-bench CLI with OpenRouter model prices registered in LiteLLM.

    python scripts/tau2_cli.py run --domain telecom ...     # same arguments as `tau2`

LiteLLM's price list knows `gpt-4.1` but not OpenRouter's names for it, so τ²-bench logs
"This model isn't mapped yet" on every message and records the cost as $0. Before starting
τ²-bench, this copies each OpenAI model's list price onto its OpenRouter aliases
(`openrouter/openai/<model>` and `openrouter/<model>`). OpenRouter passes OpenAI's prices
through, but treat these as estimates: your OpenRouter activity page shows actual spend.
"""

from __future__ import annotations

import sys

OPENAI_MODELS = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-4o", "gpt-4o-mini"]


def register_openrouter_prices() -> None:
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


def main() -> None:
    register_openrouter_prices()
    from tau2.cli import main as tau2_main

    sys.argv = ["tau2", *sys.argv[1:]]
    tau2_main()


if __name__ == "__main__":
    main()
