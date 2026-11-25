#!/usr/bin/env bash
# Stage 2b's declared test extension (STAGE2.md, amended Mon 23 Nov): v2 (S2) and S2E4 each to 8
# test samples, run side by side on one server, then the paired comparisons, then stop this pod.
# Needs the server up with both adapters ("sft2" and "sft2e4") and OPENAI_API_KEY set.
#   bash scripts/pod/extend_test.sh
set -u
cd "$(dirname "$0")/../.."
mkdir -p results
python scripts/pod/overnight.py S2 configs/sft/arms.sft2.yaml --split test --samples 8 --workers 4 > results/extend-S2.log 2>&1 &
python scripts/pod/overnight.py S2E4 configs/sft/arms.s2e4.yaml --split test --samples 8 --workers 4 > results/extend-S2E4.log 2>&1 &
wait
{
  echo "== primary: all 8 samples"
  python scripts/arms/stage2b_compare.py S2 S2E4 --split test --samples 8
  echo "== check: samples 2-7 only (both run with today's code)"
  python scripts/arms/stage2b_compare.py S2 S2E4 --split test --samples 6 --min-sample 2
} 2>&1 | grep -v DEBUG | tee results/extend-compare.txt
runpodctl stop pod "$RUNPOD_POD_ID"
