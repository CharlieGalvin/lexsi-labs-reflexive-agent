#!/usr/bin/env bash
# Arm $ARM overnight: wait for training (tmux session "h2"), serve $ARM (and SFT v2 if found), run
# validation (the rule) and dev, then the comparisons. Never test.
#
#   tmux new -d -s h2eval "bash /workspace/charlie-h2/scripts/pod/h2_overnight.sh"
#   H3: tmux new -d -s h3eval "ARM=H3 bash /workspace/charlie-h2/scripts/pod/h2_overnight.sh"
#   ONLY_VAL=1 stops after validation.
#
# Logs: /workspace/h2-eval.log (this script), /workspace/h2-serve.log (vLLM).
# Summary for the morning: /workspace/h2-results.txt
set -u
REPO=/workspace/charlie-h2
KATIE=/workspace/katie/lexsi-labs-reflexive-agent
PY=$KATIE/.venv/bin/python   # Katie's environment, with our code first on the path
export PYTHONPATH=$REPO/src
ARMS=$REPO/results/episodes/arms
ARM="${ARM:-H2}"
if [ "$ARM" = H3 ]; then
  RUN=$REPO/runs/grpo-decision-h3-aya32b-v2 CFG=configs/sft/arms.grpo_decision_h3.yaml SERVED=decision3 TRAIN=h3
else
  RUN=$REPO/runs/grpo-decision-aya32b-v2 CFG=configs/sft/arms.grpo_decision.yaml SERVED=decision TRAIN=h2
fi
LOW="$(echo "$ARM" | tr '[:upper:]' '[:lower:]')"
OUT=/workspace/$LOW-results.txt
exec >>/workspace/$LOW-eval.log 2>&1
cd "$REPO"
# shellcheck disable=SC1091
source /workspace/.pod_env 2>/dev/null
for k in HF_TOKEN OPENAI_API_KEY OPENROUTER_API_KEY; do
  v="$(tr '\0' '\n' </proc/1/environ | sed -n "s/^$k=//p")"
  [ -n "$v" ] && export "$k=$v"
done
say() { echo "=== $* ($(date +%T))"; echo "$* ($(date +%T))" >>"$OUT"; }

say "$ARM evaluation started; waiting for training (tmux $TRAIN)"
while tmux has-session -t "=$TRAIN" 2>/dev/null; do sleep 60; done
if [ ! -f "$RUN/adapter_config.json" ]; then
  say "STOPPED: no adapter in $RUN, training failed: see tail /workspace/h2-train.log"
  exit 1
fi
say "training finished: $(grep -o 'done in [^;]*' /workspace/$LOW-train.log | tail -1)"

# SFT v2's own validation and dev conversations, to compare against (never overwritten)
for d in /workspace/katie/lexsi-labs-reflexive-agent /workspace/lexsi-labs-reflexive-agent; do
  cp -n "$d"/results/episodes/arms/S2-*-val.jsonl "$d"/results/episodes/arms/S2-*-dev.jsonl "$ARMS"/ 2>/dev/null
done
ls "$ARMS"

# Serve $ARM as "decision" (plus SFT v2 as "sft2" if its adapter is on the volume)
V2="$(ls -d /workspace/*/runs/sft-aya32b-r16-v2 /workspace/runs/sft-aya32b-r16-v2 2>/dev/null | head -1)"
EXTRA="$SERVED=$RUN"
[ -n "$V2" ] && EXTRA="$EXTRA sft2=$V2"
tmux kill-session -t "=serve" 2>/dev/null  # a server from an earlier arm
sleep 5
tmux new -d -s serve "ADAPTER= EXTRA_ADAPTERS='$EXTRA' bash $KATIE/scripts/pod/serve.sh >/workspace/$LOW-serve.log 2>&1"
until curl -sf localhost:8000/v1/models >/dev/null; do
  sleep 15
  if ! tmux has-session -t "=serve" 2>/dev/null; then
    say "STOPPED: the server died: tail /workspace/$LOW-serve.log"
    exit 1
  fi
done
say "server up ($EXTRA)"

# Validation: 30 tasks x 2 samples, 4 at a time (GPT-4.1's rate limit), twice to redo crashes
for pass in 1 2; do
  $PY scripts/arms/val_rollout.py $ARM "$CFG" --workers 4
done
say "VALIDATION (the rule: total >= 19.5; no domain more than 1 below v2; Tool/Arg <= 4/60)"
$PY scripts/arms/samples.py S2 $ARM --split val | tee -a "$OUT"
$PY scripts/arms/tool_arg.py S2 $ARM --split val | tee -a "$OUT"
if [ -n "${ONLY_VAL:-}" ]; then say "ONLY_VAL: stopping after validation"; exit 0; fi

# Dev: 36 tasks x 2 samples, whatever validation says (SEL-49's question), twice to redo crashes
for pass in 1 2; do
  for d in retail airline telecom; do
    $PY scripts/arms/rollout.py $ARM "$d" --split dev --samples 2 --workers 4 --config "$CFG" \
      --out "$ARMS/$ARM-$d-dev.jsonl"
  done
done
say "DEV"
$PY scripts/arms/samples.py S2 $ARM --split dev | tee -a "$OUT"
$PY scripts/arms/tool_arg.py S2 $ARM --split dev | tee -a "$OUT"
$PY scripts/arms/gate_check.py S2 $ARM | tee -a "$OUT"
TMP=/workspace/h2-cmp && mkdir -p $TMP
for a in S2 $ARM; do cat "$ARMS"/$a-{retail,airline,telecom}-dev.jsonl >$TMP/$a-dev.jsonl; done
$PY scripts/arms/compare.py $TMP/S2-dev.jsonl $TMP/$ARM-dev.jsonl --names S2-dev $ARM-dev | tee -a "$OUT"
say "ALL DONE. The server is still up (tmux serve); stop the pod if nobody needs it."
