#!/usr/bin/env bash
# SEL-31: first end-to-end τ²-bench run + cost per task.
# Works with an OpenAI key OR an OpenRouter key (τ²-bench uses LiteLLM, which supports both).
#
#   just tau2-first-run              # 2 retail train tasks (just loads .env for you)
#   just tau2-first-run telecom      # try telecom
#
# Without just: set -a; source .env; set +a; DOMAIN=retail bash scripts/tau2_first_run.sh
# Put OPENAI_API_KEY or OPENROUTER_API_KEY in .env (copied from .env.example; never commit .env).
#
# Costs a few cents: 2 train tasks, 1 trial, GPT-4.1 as both agent and simulated customer.
set -euo pipefail

if ! python -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "τ²-bench needs Python 3.12 or 3.13; this venv is $(python -V 2>&1)." >&2
  echo "Fix: uv venv --python 3.12 .venv && source .venv/bin/activate && uv pip install -e '.[bench]'" >&2
  exit 1
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  DEFAULT_LLM=gpt-4.1-2025-04-14             # τ²-bench's default model, pinned snapshot
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  DEFAULT_LLM=openrouter/openai/gpt-4.1      # same model, routed through OpenRouter
else
  echo "Set OPENAI_API_KEY or OPENROUTER_API_KEY first (see .env.example)" >&2
  exit 1
fi
AGENT_LLM=${AGENT_LLM:-$DEFAULT_LLM}
USER_LLM=${USER_LLM:-$DEFAULT_LLM}         # the simulated customer
DOMAIN=${DOMAIN:-retail}                  # retail | airline | telecom
OUT=${OUT:-tau2-first-run-$DOMAIN}
echo "agent=$AGENT_LLM  customer=$USER_LLM  domain=$DOMAIN"

python -c "import tau2" 2>/dev/null || pip install -e ".[bench]"

echo "== 1. Plan only (no API calls)"
tau2 run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT" --dry-run

echo "== 2. Real run: 2 $DOMAIN train tasks"
tau2 run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT"

echo "== 3. Official scores, tokens and cost per task"
python "$(dirname "$0")/tau2_report.py" "$OUT"
echo "Browse the run with: tau2 view"
