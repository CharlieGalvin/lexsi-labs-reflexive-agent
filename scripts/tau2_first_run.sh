#!/usr/bin/env bash
# SEL-31: first end-to-end τ²-bench run + cost per task. Needs an OpenAI key (or any LiteLLM provider).
#
#   cp .env.example .env   # add OPENAI_API_KEY
#   set -a; source .env; set +a
#   bash scripts/tau2_first_run.sh
#
# Costs a few cents: 2 train tasks (retail by default; DOMAIN=telecom bash ... to try telecom), 1 trial, gpt-4.1 as both agent and simulated customer.
set -euo pipefail

AGENT_LLM=${AGENT_LLM:-gpt-4.1-2025-04-14}
USER_LLM=${USER_LLM:-gpt-4.1-2025-04-14}   # τ²-bench's default user simulator
DOMAIN=${DOMAIN:-retail}                  # retail | airline | telecom
OUT=${OUT:-tau2-first-run-$DOMAIN}

python -c "import tau2" 2>/dev/null || pip install -e ".[bench]"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY (see .env.example)}"

echo "== 1. Plan only (no API calls)"
tau2 run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT" --dry-run

echo "== 2. Real run: 2 $DOMAIN train tasks"
tau2 run --domain "$DOMAIN" --task-split-name train --num-tasks 2 --num-trials 1 \
  --agent-llm "$AGENT_LLM" --user-llm "$USER_LLM" --save-to "$OUT"

echo "== 3. Official scores + cost per task"
python - "$OUT" <<'EOF'
import json, pathlib, sys
root = pathlib.Path("data/simulations") / sys.argv[1]
files = [root] if root.is_file() else sorted(root.rglob("*.json"))
for f in files:
    try:
        data = json.loads(f.read_text())
    except Exception:
        continue
    for sim in data.get("simulations", []) if isinstance(data, dict) else []:
        r = (sim.get("reward_info") or {}).get("reward")
        print(f"task {sim.get('task_id')}: reward={r}  agent_cost=${sim.get('agent_cost') or 0:.4f}  "
              f"user_cost=${sim.get('user_cost') or 0:.4f}")
EOF
echo "Browse the run with: tau2 view"
