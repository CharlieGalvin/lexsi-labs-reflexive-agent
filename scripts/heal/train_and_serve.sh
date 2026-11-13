#!/usr/bin/env bash
# A heal cycle's training command (SEL-68): stop vLLM, train the adapter on the cycle's fixes, start
# vLLM again. One GPU can't train and serve at once. Run on the pod by `just heal`:
#
#   just heal --config configs/sft/arms.base.yaml --sample 10 \
#     --train-cmd "bash scripts/heal/train_and_serve.sh {data} {out} {base}" --load-adapter \
#     --out results/heal/first-cycle
#
# {data} is the cycle's fixes (reagent.sft/v1), {out} the folder the cycle reads the adapter from
# (its name is the adapter's name), {base} the adapter being healed (empty for the base student).
#
# 1. Stop vLLM: every process matching VLLM_MATCH, then wait until the GPU's memory is free.
# 2. Train: scripts/sft/train_sft.py with configs/train/sft_heal.yaml on {data}, into {out}. With a
#    {base}, training continues that adapter (--init-adapter), found next to {out} or in runs/.
# 3. Start vLLM again (scripts/pod/serve.sh, RUNTIME_LORA=1), in the background with its own log,
#    serving the {base} adapter under its name (the gate's baseline) plus HEAL_SERVE_ADAPTERS, and
#    wait until it answers. The cycle then loads the new adapter itself (--load-adapter).
#
# vLLM is started again even when training fails, so the server is back either way; the exit code
# is then training's. Settings (environment):
#   HEAL_INIT_ADAPTER    the first cycle's starting adapter when the student isn't the base model,
#                        e.g. runs/sft-aya32b-r16-v2 for configs/sft/arms.sft2.yaml (model sft2)
#   HEAL_SERVE_ADAPTERS  adapters to keep serving, "name=path ...": e.g. "sft2=runs/sft-aya32b-r16-v2"
#                        when the cycle's student is sft2 (the gate's baseline)
#   HEAL_SERVE_NEW=1     also serve the new adapter from the restart (scripts/pod/heal_service.py:
#                        the cycle runs elsewhere and reaches the pod only through its endpoint)
#   SERVE_LOG (default $WORKSPACE/vllm-heal.log), PORT (8000), READY_TIMEOUT (1200 s),
#   STOP_TIMEOUT (120 s), HEAL_TRAIN_CONFIG, and for tests HEAL_TRAIN, HEAL_SERVE, VLLM_MATCH.
#   VLLM_API_KEY, when vLLM requires a key: the readiness check sends it. $WORKSPACE/.pod_env (setup.sh's
#   caches, HF_HOME) is loaded first, as serve.sh does.

set -euo pipefail

DATA="${1:?usage: train_and_serve.sh DATA OUT [BASE]}"
OUT="${2:?usage: train_and_serve.sh DATA OUT [BASE]}"
BASE="${3-}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
# the pod's caches (setup.sh), as serve.sh loads them: HF_HOME on the volume, or training downloads the
# model again into the container's disk. Here too, since the heal service may start without them.
# shellcheck disable=SC1091
[[ -f "$WORKSPACE/.pod_env" ]] && source "$WORKSPACE/.pod_env"
PORT="${PORT:-8000}"
CONFIG="${HEAL_TRAIN_CONFIG:-configs/train/sft_heal.yaml}"
TRAIN="${HEAL_TRAIN:-$REPO_DIR/.venv/bin/python scripts/sft/train_sft.py}"
SERVE="${HEAL_SERVE:-bash scripts/pod/serve.sh}"
VLLM_MATCH="${VLLM_MATCH:-vllm serve}"
SERVE_LOG="${SERVE_LOG:-$WORKSPACE/vllm-heal.log}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
STOP_TIMEOUT="${STOP_TIMEOUT:-120}"
POLL="${POLL:-5}"
GPU_FREE_MB="${GPU_FREE_MB:-2000}"
HEAL_INIT_ADAPTER="${HEAL_INIT_ADAPTER-}"
HEAL_SERVE_ADAPTERS="${HEAL_SERVE_ADAPTERS-}"
HEAL_SERVE_NEW="${HEAL_SERVE_NEW:-0}"

log() { echo "[train_and_serve $(date +%H:%M:%S)] $*"; }
die() { echo "⛔ $*" >&2; exit 1; }

cd "$REPO_DIR"
[[ -f "$DATA" ]] || die "no fixes file at $DATA"
NAME="$(basename "$OUT")"

# --- Where the adapter being healed lives -----------------------------------------------------
init=""
if [[ -n "$BASE" ]]; then
  for d in "$(dirname "$OUT")/$BASE" "runs/$BASE"; do
    if [[ -f "$d/adapter_config.json" ]]; then init="$d"; break; fi
  done
  [[ -n "$init" ]] || die "no adapter for base $BASE: looked in $(dirname "$OUT")/$BASE and runs/$BASE"
elif [[ -n "$HEAL_INIT_ADAPTER" ]]; then
  [[ -f "$HEAL_INIT_ADAPTER/adapter_config.json" ]] || die "no adapter in HEAL_INIT_ADAPTER=$HEAL_INIT_ADAPTER"
  init="$HEAL_INIT_ADAPTER"
fi

# --- 1. Stop vLLM -------------------------------------------------------------------------------
pids="$(pgrep -f -- "$VLLM_MATCH" || true)"
if [[ -n "$pids" ]]; then
  log "stopping vLLM (pid $(echo "$pids" | tr '\n' ' '))"
  # shellcheck disable=SC2086 # one pid per word
  kill $pids 2>/dev/null || true
  deadline=$(( SECONDS + STOP_TIMEOUT ))
  while pgrep -f -- "$VLLM_MATCH" >/dev/null; do
    if (( SECONDS >= deadline )); then
      log "still running after ${STOP_TIMEOUT}s: killing"
      pkill -9 -f -- "$VLLM_MATCH" || true
      break
    fi
    sleep "$POLL"
  done
else
  log "no vLLM running (nothing matches '$VLLM_MATCH')"
fi
if command -v nvidia-smi >/dev/null; then  # its worker processes can hold memory a little longer
  deadline=$(( SECONDS + STOP_TIMEOUT ))
  until (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) < GPU_FREE_MB )); do
    (( SECONDS < deadline )) || die "the GPU still has over ${GPU_FREE_MB} MiB in use: is something else on it?"
    sleep "$POLL"
  done
fi

# --- 2. Train ----------------------------------------------------------------------------------
read -ra train_cmd <<< "$TRAIN"
args=("$CONFIG" --data "$DATA" --out "$OUT" --name "$NAME")
[[ -n "$init" ]] && args+=(--init-adapter "$init")
log "training $NAME on $(wc -l < "$DATA" | tr -d ' ') fixes${init:+, from $init}"
train_status=0
"${train_cmd[@]}" "${args[@]}" || train_status=$?
if (( train_status == 0 )) && [[ ! -f "$OUT/adapter_config.json" ]]; then
  log "training exited 0 but wrote no adapter to $OUT"
  train_status=1
fi
if (( train_status == 0 )); then
  log "adapter written to $OUT"
else
  log "training failed (exit $train_status)"
fi

# --- 3. Start vLLM again -------------------------------------------------------------------------
extras="$HEAL_SERVE_ADAPTERS"
[[ -n "$BASE" ]] && extras="${extras:+$extras }$BASE=$init"
# HEAL_SERVE_NEW=1 (the pod heal service): serve the new adapter from the start too, so a cycle
# running elsewhere, which can't load a file on the pod, finds it on the endpoint
[[ "$HEAL_SERVE_NEW" == 1 && "$train_status" == 0 ]] && extras="${extras:+$extras }$NAME=$OUT"
read -ra serve_cmd <<< "$SERVE"
mkdir -p "$(dirname "$SERVE_LOG")"
log "starting vLLM on :$PORT (runtime LoRA${extras:+, serving $extras}); log: $SERVE_LOG"
# its own log and no stdin: the cycle waits for this command's output to close, and must not wait
# for the server's. setsid (Linux) keeps it out of the cycle's process group.
detach=()
command -v setsid >/dev/null && detach=(setsid)
ADAPTER="" EXTRA_ADAPTERS="$extras" RUNTIME_LORA=1 PORT="$PORT" \
  nohup ${detach[@]+"${detach[@]}"} "${serve_cmd[@]}" >"$SERVE_LOG" 2>&1 </dev/null &
server=$!

# a vLLM started with VLLM_API_KEY answers 401 without it
auth=()
[[ -n "${VLLM_API_KEY-}" ]] && auth=(-H "Authorization: Bearer $VLLM_API_KEY")
deadline=$(( SECONDS + READY_TIMEOUT ))
until curl -sf ${auth[@]+"${auth[@]}"} "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
  if ! kill -0 "$server" 2>/dev/null && ! pgrep -f -- "$VLLM_MATCH" >/dev/null; then
    tail -20 "$SERVE_LOG" >&2 || true
    die "vLLM exited while starting: see $SERVE_LOG"
  fi
  (( SECONDS < deadline )) || die "vLLM isn't answering on :$PORT after ${READY_TIMEOUT}s: see $SERVE_LOG"
  sleep "$POLL"
done
log "vLLM is up on :$PORT"
exit "$train_status"
