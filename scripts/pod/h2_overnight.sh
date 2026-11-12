#!/usr/bin/env bash
# Arm H2 overnight: wait for training (tmux session "h2"), serve H2 (and SFT v2 if found), run
# validation (the rule) and dev, then the comparisons. Never test.
#
#   tmux new -d -s h2eval "bash /workspace/charlie-h2/scripts/pod/h2_overnight.sh"
#
# Logs: /workspace/h2-eval.log (this script), /workspace/h2-serve.log (vLLM).
# Summary for the morning: /workspace/h2-results.txt
set -u
REPO=/workspace/charlie-h2
KATIE=/workspace/katie/lexsi-labs-reflexive-agent
PY=$KATIE/.venv/bin/python   # Katie's environment, with our code first on the path
export PYTHONPATH=$REPO/src
ARMS=$REPO/results/episodes/arms
RUN=$REPO/runs/grpo-decision-aya32b-v2
CFG=configs/sft/arms.grpo_decision.yaml
OUT=/workspace/h2-results.txt
exec >>/workspace/h2-eval.log 2>&1
cd "$REPO"
# shellcheck disable=SC1091
source /workspace/.pod_env 2>/dev/null
for k in HF_TOKEN OPENAI_API_KEY OPENROUTER_API_KEY; do
  v="$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$k=//p")"
  [ -n "$v" ] && export "$k=$v"
done
say() { echo "=== $* ($(date +%T))"; echo "$* ($(date +%T))" >>"$OUT"; }

say "H2 overnight started; waiting for training (tmux h2)"
while tmux has-session -t h2 2>/dev/null; do sleep 60; done
if [ ! -f "$RUN/adapter_config.json" ]; then
  say "STOPPED: no adapter in $RUN, training failed: see tail /workspace/h2-train.log"
  exit 1
fi
say "training finished: $(grep -o 'train_runtime[^,]*' /workspace/h2-train.log | tail -1)"

# SFT v2's own validation and dev conversations, to compare against (never overwritten)
for d in /workspace/katie/lexsi-labs-reflexive-agent /workspace/lexsi-labs-reflexive-agent; do
  cp -n "$d"/results/episodes/arms/S2-*-val.jsonl "$d"/results/episodes/arms/S2-*-dev.jsonl "$ARMS"/ 2>/dev/null
done
ls "$ARMS"

# Serve H2 as "decision" (plus SFT v2 as "sft2" if its adapter is on the volume)
V2="$(ls -d /workspace/*/runs/sft-aya32b-r16-v2 /workspace/runs/sft-aya32b-r16-v2 2>/dev/null | head -1)"
EXTRA="decision=$RUN"
[ -n "$V2" ] && EXTRA="$EXTRA sft2=$V2"
tmux new -d -s serve "ADAPTER= EXTRA_ADAPTERS='$EXTRA' bash $KATIE/scripts/pod/serve.sh >/workspace/h2-serve.log 2>&1"
until curl -sf localhost:8000/v1/models >/dev/null; do
  sleep 15
  if ! tmux has-session -t serve 2>/dev/null; then
    say "STOPPED: the server died: tail /workspace/h2-serve.log"
    exit 1
  fi
done
say "server up ($EXTRA)"

# Validation: 30 tasks x 2 samples, 4 at a time (GPT-4.1's rate limit), twice to redo crashes
for pass in 1 2; do
  $PY scripts/arms/val_rollout.py H2 "$CFG" --workers 4
done
say "VALIDATION (the rule: total >= 19.5; no domain more than 1 below v2; Tool/Arg <= 4/60)"
$PY scripts/arms/samples.py S2 H2 --split val | tee -a "$OUT"
$PY scripts/arms/tool_arg.py S2 H2 --split val | tee -a "$OUT"

# Dev: 36 tasks x 2 samples, whatever validation says (SEL-49's question), twice to redo crashes
for pass in 1 2; do
  for d in retail airline telecom; do
    $PY scripts/arms/rollout.py H2 "$d" --split dev --samples 2 --workers 4 --config "$CFG" \
      --out "$ARMS/H2-$d-dev.jsonl"
  done
done
say "DEV"
$PY scripts/arms/samples.py S2 H2 --split dev | tee -a "$OUT"
$PY scripts/arms/tool_arg.py S2 H2 --split dev | tee -a "$OUT"
$PY scripts/arms/gate_check.py S2 H2 | tee -a "$OUT"
TMP=/workspace/h2-cmp && mkdir -p $TMP
for a in S2 H2; do cat "$ARMS"/$a-{retail,airline,telecom}-dev.jsonl >$TMP/$a-dev.jsonl; done
$PY scripts/arms/compare.py $TMP/S2-dev.jsonl $TMP/H2-dev.jsonl --names S2-dev H2-dev | tee -a "$OUT"
say "ALL DONE. The server is still up (tmux serve); stop the pod if nobody needs it."
