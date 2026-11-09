#!/usr/bin/env bash
# Serve Aya Expanse 32B with vLLM on the pod, with the SFT adapter loaded (SEL-65; reusable for
# SEL-63). OpenAI-compatible, on port 8000, so the rollout harness talks to it like Cohere's API.
#
#   bash scripts/pod/serve.sh                          # base model + runs/sft-aya32b-r16 as "sft"
#   ADAPTER=runs/other bash scripts/pod/serve.sh       # another adapter
#   ADAPTER= bash scripts/pod/serve.sh                 # the base model only
#   ADAPTER= EXTRA_ADAPTERS="sft2=runs/sft-aya32b-r16-v2" bash scripts/pod/serve.sh   # v2 only
#   EXTRA_ADAPTERS="sft2=runs/sft-aya32b-r16-v2" bash scripts/pod/serve.sh   # v1 as "sft" + v2 as "sft2"
#   ADAPTER= RUNTIME_LORA=1 bash scripts/pod/serve.sh  # heal cycles: adapters loaded while serving
#   VLLM_API_KEY=... (read by vLLM itself): require a key, when the port is public (configs/sft/arms.pod.yaml)
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
EXTRA_ADAPTERS="${EXTRA_ADAPTERS-}"   # more adapters on the same server: "name=path name=path"
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

# Every adapter to load: ADAPTER as "sft" (unless empty), plus EXTRA_ADAPTERS ("name=path ...")
mods=()
[[ -n "$ADAPTER" ]] && mods=("sft=$ADAPTER")
for m in $EXTRA_ADAPTERS; do
  [[ "$m" == *=* ]] || { echo "⛔ EXTRA_ADAPTERS entry '$m' isn't name=path" >&2; exit 1; }
  mods+=("$m")
done
# RUNTIME_LORA=1 (heal cycles, SEL-68): LoRA on even with no adapter at start, and adapters can be
# loaded while serving (POST /v1/load_lora_adapter, `just heal --load-adapter`). MAX_LORA_RANK and
# MAX_LORAS then size the slots for adapters that aren't known yet.
RUNTIME_LORA="${RUNTIME_LORA:-0}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
MAX_LORAS="${MAX_LORAS:-4}"
lora=()
if (( ${#mods[@]} )) || [[ "$RUNTIME_LORA" == 1 ]]; then
  rank=0  # vLLM needs the largest rank among them, not the first one's
  for m in ${mods[@]+"${mods[@]}"}; do  # (an empty array, safely, on macOS's bash 3 too)
    dir="${m#*=}"
    [[ -f "$dir/adapter_config.json" ]] || { echo "⛔ no adapter in $dir: train it first (just sft-train)" >&2; exit 1; }
    r="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['r'])" "$dir/adapter_config.json")"
    (( r > rank )) && rank="$r"
  done
  slots="${#mods[@]}"
  if [[ "$RUNTIME_LORA" == 1 ]]; then
    export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
    (( MAX_LORA_RANK > rank )) && rank="$MAX_LORA_RANK"
    (( MAX_LORAS > slots )) && slots="$MAX_LORAS"
  fi
  lora=(--enable-lora)
  (( ${#mods[@]} )) && lora+=(--lora-modules "${mods[@]}")
  lora+=(--max-lora-rank "$rank" --max-loras "$slots")
  echo "serving $MODEL + ${mods[*]:-no adapter yet} (max rank $rank, $slots slots, runtime loading: $RUNTIME_LORA) on :$PORT, context $MAX_LEN"
  [[ -z "$ADAPTER" && "$RUNTIME_LORA" != 1 ]] && echo "   note: no \"sft\" adapter (ADAPTER is empty): arm S won't find its model"
else
  echo "serving $MODEL (no adapter) on :$PORT, context $MAX_LEN"
fi

# Aya Expanse 32B's config.json says max_position_embeddings=8192, but the model serves 128K
# (Cohere's API runs it at 128K, as for Arm A; RoPE with a large base). vLLM sizes its RoPE table
# from that number, so VLLM_ALLOW_LONG_MAX_MODEL_LEN alone let conversations past 8,192 tokens index
# off the end of it: "CUDA error: device-side assert triggered", and the server died mid-run.
# Telling vLLM the real length builds the table to MAX_LEN instead.

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
  --hf-overrides "{\"max_position_embeddings\": $MAX_LEN}" \
  "${lora[@]}"
