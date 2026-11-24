#!/usr/bin/env bash
# Stage 2c on pod B, unattended (STAGE2.md, amended Tue 24 Nov; SEL-121), then stop the pod:
# train S2P1E4 (round 1's data, 4 epochs), serve it, validation at 8 samples, test at 8 samples.
# Every step logs to results/night-b/<step>.log.
#   bash scripts/pod/night_b.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO=$PWD
# A new tmux window can start without the repo's environment: use it explicitly.
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
LOG=results/night-b
mkdir -p "$LOG"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG/queue.log"; }
ready() {
  for _ in $(seq 1 100); do curl -s localhost:8000/v1/models | grep -q "\"$1\"" && return 0; sleep 15; done
  return 1
}
[ -n "${OPENAI_API_KEY:-}" ] || { say "⛔ OPENAI_API_KEY is empty"; exit 1; }
python -c "import reflexive, tau2" 2>> "$LOG/queue.log" || { say "⛔ python can't import reflexive/tau2: $(command -v python)"; exit 1; }
say "python: $(command -v python)"
pkill -f "vllm serve"; sleep 20

if [ ! -f runs/sft-aya32b-p1-e4/adapter_config.json ]; then
  say "training S2P1E4"
  python scripts/sft/train_sft.py configs/train/sft_aya32b_p1_e4.yaml --push > "$LOG/train.log" 2>&1 \
    || { say "⛔ training failed: see $LOG/train.log"; runpodctl stop pod "$RUNPOD_POD_ID"; exit 1; }
fi

ADAPTER= EXTRA_ADAPTERS="sft2p1e4=$REPO/runs/sft-aya32b-p1-e4" bash scripts/pod/serve.sh > "$LOG/serve.log" 2>&1 &
if ready sft2p1e4; then
  say "S2P1E4 on validation, 8 samples"
  for i in 1 2 3; do
    python scripts/arms/val_rollout.py S2P1E4 configs/sft/arms.s2p1e4.yaml --from results/episodes/arms \
      --samples 8 --actions off >> "$LOG/val.log" 2>&1 && break
    sleep 300
  done
  python scripts/arms/stage2b_compare.py S2 S2P1E4 > "$LOG/val-compare-v2.txt" 2>&1
  say "S2P1E4 on test, 8 samples"
  python scripts/pod/overnight.py S2P1E4 configs/sft/arms.s2p1e4.yaml --split test --samples 8 --workers 6 \
    > "$LOG/test.log" 2>&1
else
  say "⛔ the server didn't come up: see $LOG/serve.log"
fi
say "queue done; stopping the pod"
runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG/queue.log" 2>&1
