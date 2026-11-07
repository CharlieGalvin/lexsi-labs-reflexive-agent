#!/usr/bin/env bash
# Serve Aya Expanse 32B with vLLM on the pod, with the SFT adapter loaded (SEL-65; reusable for
# SEL-63). OpenAI-compatible, on port 8000, so the rollout harness talks to it like Cohere's API.
#
#   bash scripts/pod/serve.sh                          # base model + runs/sft-aya32b-r16 as "sft"
#   ADAPTER=runs/other bash scripts/pod/serve.sh       # another adapter
#   ADAPTER= bash scripts/pod/serve.sh                 # the base model only
#
# Then, in another terminal on the pod, the rollouts (configs/sft/: P = base, S = adapter):
#   just rollout P retail --split dev --limit 5 --config configs/sft/arms.base.yaml
#   just rollout S retail --split dev --config configs/sft/arms.sft.yaml
#
# The GPU can't train and serve at once: stop training first. The 16-bit weights take ~64 GB,
# which leaves room for a MAX_LEN-token context (default 32768) on an 80 GB card. Conversations
# longer than that are cut off; `just arm-metrics` counts them.

set -euo pipefail

MODEL="${MODEL:-CohereLabs/aya-expanse-32b}"
ADAPTER="${ADAPTER-runs/sft-aya32b-r16}"
MAX_LEN="${MAX_LEN:-32768}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.92}"
WORKSPACE="${WORKSPACE:-/workspace}"
VLLM="${VLLM:-$WORKSPACE/vllm-env/bin/vllm}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# shellcheck disable=SC1091
[[ -f "$WORKSPACE/.pod_env" ]] && source "$WORKSPACE/.pod_env"
[[ -x "$VLLM" ]] || { echo "⛔ no vLLM at $VLLM: run scripts/pod/setup.sh first" >&2; exit 1; }
cd "$REPO_DIR"

# Aya's template refuses a conversation that starts with the agent's greeting, as every τ²-bench
# one does; serve the same template training used, without that one check (reflexive.sft)
TEMPLATE="$WORKSPACE/chat_template.$(basename "$MODEL").jinja"
"$REPO_DIR/.venv/bin/python" scripts/sft/chat_template.py "$MODEL" "$TEMPLATE"

lora=()
if [[ -n "$ADAPTER" ]]; then
  [[ -f "$ADAPTER/adapter_config.json" ]] || { echo "⛔ no adapter in $ADAPTER: train it first (just sft-train)" >&2; exit 1; }
  rank="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['r'])" "$ADAPTER/adapter_config.json")"
  lora=(--enable-lora --lora-modules "sft=$ADAPTER" --max-lora-rank "$rank")
  echo "serving $MODEL + adapter $ADAPTER (rank $rank) as \"sft\" on :$PORT, context $MAX_LEN"
else
  echo "serving $MODEL (no adapter) on :$PORT, context $MAX_LEN"
fi

exec "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config vllm \
  --chat-template "$TEMPLATE" \
  "${lora[@]}"
