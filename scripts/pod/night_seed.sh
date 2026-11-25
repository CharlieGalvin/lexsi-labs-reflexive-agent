#!/usr/bin/env bash
# Stage 2d on one pod, unattended (STAGE2.md, amended Wed 25 Nov), then stop the pod:
# train S2E4 again with seed 1 (S2E4S1), serve it, test at 8 samples, then the paired comparisons
# against v2 (the pre-registered rule) and against S2E4 (reported beside it).
# Every step logs to results/night-seed/<step>.log.
#   bash scripts/pod/night_seed.sh
set -u
cd "$(dirname "$(readlink -f "$0")")/../.."
REPO=$PWD
# A new tmux window can start without the repo's environment: use it explicitly.
# shellcheck disable=SC1091
source "$REPO/.venv/bin/activate"
LOG=results/night-seed
mkdir -p "$LOG"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG/queue.log"; }
stop() { say "stopping the pod"; runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOG/queue.log" 2>&1; }
ready() {
  for _ in $(seq 1 100); do curl -s localhost:8000/v1/models | grep -q "\"$1\"" && return 0; sleep 15; done
  return 1
}
[ -n "${OPENAI_API_KEY:-}" ] || { say "⛔ OPENAI_API_KEY is empty"; exit 1; }
python -c "import reflexive, tau2" 2>> "$LOG/queue.log" || { say "⛔ python can't import reflexive/tau2: $(command -v python)"; exit 1; }
for d in retail airline telecom; do
  for a in S2 S2E4; do
    [ -f "results/episodes/arms/$a-$d-test.jsonl" ] || say "⚠ results/episodes/arms/$a-$d-test.jsonl is missing: copy it here before the comparisons"
  done
done
say "python: $(command -v python)"
pkill -f "vllm serve"; sleep 20

if [ ! -f runs/sft-aya32b-v2-e4-s1/adapter_config.json ]; then
  say "training S2E4S1 (seed 1)"
  python scripts/sft/train_sft.py configs/train/sft_aya32b_v2_e4_s1.yaml --push > "$LOG/train.log" 2>&1 \
    || { say "⛔ training failed: see $LOG/train.log"; stop; exit 1; }
fi

ADAPTER= EXTRA_ADAPTERS="sft2e4s1=$REPO/runs/sft-aya32b-v2-e4-s1" bash scripts/pod/serve.sh > "$LOG/serve.log" 2>&1 &
if ready sft2e4s1; then
  say "S2E4S1 on test, 8 samples"
  python scripts/pod/overnight.py S2E4S1 configs/sft/arms.s2e4s1.yaml --split test --samples 8 --workers 6 \
    > "$LOG/test.log" 2>&1
  {
    echo "== primary (the pre-registered rule): S2E4S1 against v2, test, 8 samples"
    python scripts/arms/stage2b_compare.py S2 S2E4S1 --split test --samples 8
    echo "== reported beside it: S2E4S1 against S2E4 (seed 0), test, 8 samples"
    python scripts/arms/stage2b_compare.py S2E4 S2E4S1 --split test --samples 8
  } 2>&1 | grep -v DEBUG | tee "$LOG/compare.txt"
else
  say "⛔ the server didn't come up: see $LOG/serve.log"
fi
say "queue done"
stop
