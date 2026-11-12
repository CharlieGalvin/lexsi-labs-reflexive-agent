#!/usr/bin/env bash
# After the H2 overnight evaluation (tmux session "h2eval") ends, for whatever reason: upload the
# results to the adapter's Hugging Face repo (eval/), then stop this pod. The network volume and
# everything in /workspace stay.
#
#   tmux new -d -s autostop "bash /workspace/charlie-h2/scripts/pod/h2_autostop.sh"
#
# Log: /workspace/h2-autostop.log. Cancel before it fires: tmux kill-session -t autostop
set -u
REPO=/workspace/charlie-h2
PY=/workspace/katie/lexsi-labs-reflexive-agent/.venv/bin/python
HUB=reagent-lexsi/grpo-decision-aya32b-v2
exec >>/workspace/h2-autostop.log 2>&1
for k in HF_TOKEN RUNPOD_API_KEY RUNPOD_POD_ID; do
  v="$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$k=//p")"
  [ -n "$v" ] && export "$k=$v"
done
echo "=== waiting for h2eval ($(date +%T))"
while tmux has-session -t "=h2eval" 2>/dev/null; do sleep 60; done
echo "=== h2eval ended ($(date +%T)); uploading"

STAGE=/workspace/h2-upload && rm -rf "$STAGE" && mkdir -p "$STAGE"
cp /workspace/h2-results.txt /workspace/h2-eval.log "$STAGE"/ 2>/dev/null
tail -n 400 /workspace/h2-train.log >"$STAGE/h2-train.tail.log" 2>/dev/null
tail -n 200 /workspace/h2-serve.log >"$STAGE/h2-serve.tail.log" 2>/dev/null
cp "$REPO"/results/episodes/arms/H2-*.jsonl "$STAGE"/ 2>/dev/null
cp "$REPO"/results/*H2*.json "$REPO"/results/gate-S2-vs-H2-*.json "$STAGE"/ 2>/dev/null
cp "$REPO"/runs/grpo-decision-aya32b-v2/run_summary.json "$STAGE"/ 2>/dev/null
ls -la "$STAGE"
"$PY" - "$STAGE" "$HUB" <<'EOF' || echo "⚠ upload failed: the files are still in /workspace/h2-upload"
import sys
from huggingface_hub import HfApi

stage, repo = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
info = api.upload_folder(folder_path=stage, repo_id=repo, path_in_repo="eval", commit_message="H2 overnight evaluation")
print("uploaded:", info)
EOF

echo "=== stopping the pod ($(date +%T))"
if command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  runpodctl stop pod "$RUNPOD_POD_ID"
else
  echo "⚠ runpodctl or RUNPOD_POD_ID missing: can't stop the pod from inside; stop it in the console"
fi
