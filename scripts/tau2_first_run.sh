#!/usr/bin/env bash
# SEL-31: first end-to-end τ²-bench run + cost per task.
#
#   just tau2-first-run              # 2 retail train tasks (just loads .env for you)
#   just tau2-first-run telecom      # telecom: works with an OpenRouter key alone
#
# Without just: set -a; source .env; set +a; DOMAIN=retail bash scripts/tau2_first_run.sh
#
# Keys (in .env, copied from .env.example; never commit .env):
#   - Agent + simulated customer: OPENAI_API_KEY or OPENROUTER_API_KEY.
#   - Scoring: retail and airline are partly graded by LLM judges that τ²-bench pins to OpenAI
#     models, so those two domains need OPENAI_API_KEY. Telecom is graded by state checks only.
#
# Costs a few cents: 2 train tasks, 1 trial, GPT-4.1 as both agent and simulated customer.
set -euo pipefail

fail() { echo "✗ $*" >&2; exit 1; }

python -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)' ||
  fail "τ²-bench needs Python 3.12 or 3.13; this venv is $(python -V 2>&1). Fix: just setup-dev"

python - <<'PY' 2>/dev/null || fail "τ²-bench or its data folder is missing. Fix: just setup-dev (or just tau2)"
import sys
from tau2.utils.utils import DATA_DIR
sys.exit(0 if (DATA_DIR / "tau2" / "domains").exists() else 1)
PY

DOMAIN=${DOMAIN:-retail}                  # retail | airline | telecom
if [[ "$DOMAIN" != "telecom" && -z "${OPENAI_API_KEY:-}" ]]; then
  fail "$DOMAIN is scored partly by LLM judges pinned to OpenAI models (gpt-5.4-mini / gpt-5.5),
  so it needs OPENAI_API_KEY in .env. Options:
    - add an OpenAI key (only the short judge calls use it; agent + customer can stay on OpenRouter), or
    - try telecom now, which needs no judge:  just tau2-first-run telecom"
fi

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  DEFAULT_LLM=gpt-4.1-2025-04-14             # τ²-bench's default model, pinned snapshot
elif [[ -n "${OPENROUTER_API_KEY:-}" ]]; then
  DEFAULT_LLM=openrouter/openai/gpt-4.1      # same model, routed through OpenRouter
else
  fail "Set OPENAI_API_KEY or OPENROUTER_API_KEY in .env (see .env.example)"
fi
AGENT_LLM=${AGENT_LLM:-$DEFAULT_LLM}
USER_LLM=${USER_LLM:-$DEFAULT_LLM}         # the simulated customer
OUT=${OUT:-tau2-first-run-$DOMAIN}
echo "agent=$AGENT_LLM  customer=$USER_LLM  domain=$DOMAIN"

echo "== 1. Plan only (no API calls)"
python "$(dirname "$0")/tau2_cli.py" run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT" --dry-run

echo "== 2. Real run: 2 $DOMAIN train tasks"
python "$(dirname "$0")/tau2_cli.py" run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT"

echo "== 3. Official scores, tokens and cost per task"
python "$(dirname "$0")/tau2_report.py" "$OUT"
echo "Browse the run with: tau2 view"
