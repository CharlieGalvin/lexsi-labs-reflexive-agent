#!/usr/bin/env bash
# Stage 2c on pod A, unattended (STAGE2.md, amended Tue 24 Nov), then stop the pod. In order:
#   no GPU, alongside: the consent check (S2E4 vs v2, test) and the hybrid judge on v2's and
#     S2E4's failed validation conversations;
#   16-bit, 32K, serving base Aya and S2E4: S2E4 on the train tasks (1 sample) and Arm P (base Aya
#     on our server) on test (100 x 2), side by side; then retention on S2E4;
#   FP8, 64K, serving S2E4: S2E4Q on validation, 8 samples, and its comparison with S2E4.
# Every step logs to results/night-a/<step>.log; a failed step is logged and the queue moves on.
#   VALDIR=/workspace/katie/lexsi-labs-reflexive-agent/results/episodes/arms bash scripts/pod/night_a.sh
# RETENTION=0 skips retention (run elsewhere); ARM_P=0 skips Arm P on test.
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO=$PWD
# A new tmux window can start without the repo's environment: use it explicitly.
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
OLD=/workspace/charlie/reagent
RETENTION=${RETENTION:-1}
ARM_P=${ARM_P:-1}
VALDIR=${VALDIR:-/workspace/katie/lexsi-labs-reflexive-agent/results/episodes/arms}
LOG=results/night-a
mkdir -p "$LOG" results/episodes/arms
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG/queue.log"; }
ready() {  # wait for the server to list $1 (up to 25 minutes)
  for _ in $(seq 1 100); do curl -s localhost:8000/v1/models | grep -q "\"$1\"" && return 0; sleep 15; done
  return 1
}
stop_server() {
  pkill -f "vllm serve" ; sleep 30
  for _ in $(seq 1 20); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-0}" -lt 2000 ] && return 0; sleep 15
  done
}
[ -n "${OPENAI_API_KEY:-}" ] || { say "⛔ OPENAI_API_KEY is empty"; exit 1; }
python -c "import reflexive, tau2" 2>> "$LOG/queue.log" || { say "⛔ python can't import reflexive/tau2: $(command -v python)"; exit 1; }
say "python: $(command -v python)"

# The conversations scored in the frozen folder (v2 and S2E4 on validation, v2's train pool).
for f in "$OLD"/results/episodes/arms/S2-*-val.jsonl "$OLD"/results/episodes/arms/S2E4-*-val.jsonl \
         "$OLD"/results/episodes/arms/S2-*-trainpool.jsonl; do
  [ -f "$f" ] && cp -n "$f" results/episodes/arms/
done
say "copied the frozen folder's validation and train-pool files"

# No GPU: the consent check and the hybrid judge's breakdown, in the background.
(
  python scripts/arms/consent_check.py S2 S2E4 --split test > "$LOG/consent.log" 2>&1
  for arm in S2 S2E4; do
    python scripts/labels/hybrid_judge.py --arm "$arm" --split val --include-val --failures-only \
      --from "$VALDIR" > "$LOG/hybrid-$arm.log" 2>&1
  done
) &
API_JOBS=$!

# 16-bit, 32K: base Aya (as its HF name) and S2E4.
ADAPTER= EXTRA_ADAPTERS="sft2e4=$REPO/runs/sft-aya32b-v2-e4" bash scripts/pod/serve.sh > "$LOG/serve-bf16.log" 2>&1 &
if ready sft2e4; then
  say "serving 16-bit; S2E4 on the train pool and Arm P on test"
  python scripts/heal/practice.py run --arm S2E4 --config configs/sft/arms.s2e4.yaml --samples 1 \
    --from "$VALDIR" > "$LOG/s2e4-trainpool.log" 2>&1 &
  TRAINPOOL=$!
  if [ "$ARM_P" = 1 ]; then
    POD_STUDENT_URL=http://localhost:8000/v1 python scripts/pod/overnight.py P configs/sft/arms.base.yaml \
      --split test --samples 2 --retries 2 --wait 300 > "$LOG/p-test.log" 2>&1
  fi
  wait "$TRAINPOOL"
  if [ "$RETENTION" = 1 ]; then
    python - <<'PY' 2>/dev/null || pip install -q "auditkit[lmeval]" "lm-eval[api,ifeval]" > "$LOG/retention-install.log" 2>&1
import auditkit.lmeval_engine, lm_eval  # noqa
PY
    say "retention on S2E4"
    python scripts/retention/run.py sft2e4 > "$LOG/retention.log" 2>&1 || say "retention failed: see $LOG/retention.log"
  fi
else
  say "⛔ the 16-bit server didn't come up: see $LOG/serve-bf16.log"
fi
stop_server

# FP8, 64K: S2E4 only.
ADAPTER= EXTRA_ADAPTERS="sft2e4=$REPO/runs/sft-aya32b-v2-e4" QUANTIZATION=fp8 MAX_LEN=65536 \
  bash scripts/pod/serve.sh > "$LOG/serve-fp8.log" 2>&1 &
if ready sft2e4; then
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$LOG/fp8-memory.txt"
  say "serving FP8 64K; S2E4Q on validation, 8 samples"
  for i in 1 2 3; do
    python scripts/arms/val_rollout.py S2E4Q configs/sft/arms.s2e4q.yaml --from "$VALDIR" --samples 8 \
      --actions off >> "$LOG/s2e4q-val.log" 2>&1 && break
    sleep 300
  done
  python scripts/arms/stage2b_compare.py S2E4 S2E4Q > "$LOG/s2e4q-compare.txt" 2>&1
else
  say "⛔ the FP8 server didn't come up: see $LOG/serve-fp8.log"
fi
stop_server

wait "$API_JOBS"
say "queue done; stopping the pod"
runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG/queue.log" 2>&1
