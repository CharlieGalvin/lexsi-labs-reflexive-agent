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
# The GPU can't train and serve at once: stop training first. The 16-bit weights take ~60 GiB,
# which leaves room for a MAX_LEN-token context (default 32768) on an 80 GB card. Conversations
# longer than that are cut off; `just arm-metrics` counts them.

set -euo pipefail

MODEL="${MODEL:-CohereLabs/aya-expanse-32b}"
ADAPTER="${ADAPTER-runs/sft-aya32b-r16}"
MAX_LEN="${MAX_LEN:-32768}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.95}"  # 0.92 left 4.9 GiB of KV cache: under one 32K conversation
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

# Aya Expanse 32B's config.json says max_position_embeddings=8192, but the model serves 128K
# (Cohere's API runs it at 128K, as for Arm A; RoPE, a large base): vLLM refuses a longer
# max-model-len without this. Training examples reach 13.7K tokens, dev conversations more.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# FlashInfer compiles its top-k/top-p sampler on first use, which needs ninja and a CUDA
# compiler; the pod has neither on PATH ("No such file or directory: 'ninja'" at engine start).
# PyTorch's sampler is as correct, and temperature 0 doesn't sample anyway.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# vLLM's compiled graphs, kept on the volume: a restarted pod reuses them instead of compiling again
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$WORKSPACE/.cache/vllm}"

exec "$VLLM" serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config vllm \
  --chat-template "$TEMPLATE" \
  "${lora[@]}"
