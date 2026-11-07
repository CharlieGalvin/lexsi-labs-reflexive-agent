#!/usr/bin/env bash
# Set up a RunPod GPU pod for the SFT run (SEL-65, SEL-67), and for serving Aya with vLLM (SEL-63).
#
# On a fresh pod (set HF_TOKEN and OPENAI_API_KEY as the pod's environment variables first):
#
#   cd /workspace && git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent.git
#   cd lexsi-labs-reflexive-agent && git checkout <branch>     # only if it isn't on main yet
#   bash scripts/pod/setup.sh
#
# It checks the GPU (a full 80 GB card, not a MIG slice) and the disk; keeps every cache under
# /workspace, so a pod restart doesn't lose them (docs/experiments/MESSY_LOOKUP.md); installs uv
# and just; updates the repo; installs the training environment (`just setup` + the `sft` extra);
# installs vLLM in its own environment, so its pinned torch can't clash with training; checks
# HF_TOKEN and access to the gated model; then downloads Aya Expanse 32B (~65 GB) and golden
# trajectories at tag v1.1, and builds the training examples (`just train-data`).
#
# Safe to re-run: every step skips what's already done. Settings (environment variables):
#   BRANCH (default: the branch the repo is on)  MODEL=CohereLabs/aya-expanse-32b  DATASET_REV=v1.1
#   SKIP_VLLM=0  SKIP_MODEL=0

set -euo pipefail

MODEL="${MODEL:-CohereLabs/aya-expanse-32b}"
DATASET="${DATASET:-reagent-lexsi/tau2-golden-trajectories}"
DATASET_REV="${DATASET_REV:-v1.1}"
SKIP_VLLM="${SKIP_VLLM:-0}"
SKIP_MODEL="${SKIP_MODEL:-0}"
MIN_GPU_GB="${MIN_GPU_GB:-75}"
MIN_DISK_GB="${MIN_DISK_GB:-150}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Stay on the branch the repo is on unless told otherwise: checking out another branch would swap
# this very script (and its extras and recipes) out from under the running shell
BRANCH="${BRANCH:-$(git -C "$REPO_DIR" branch --show-current)}"
VLLM_ENV="$WORKSPACE/vllm-env"

step() { printf '\n== %s\n' "$*"; }
fail() { printf '⛔ %s\n' "$*" >&2; exit 1; }
ok() { printf '   ✓ %s\n' "$*"; }

step "GPU"
command -v nvidia-smi >/dev/null || fail "no nvidia-smi: is this a GPU pod?"
if nvidia-smi -L | grep -q MIG; then
  fail "this is a MIG slice: use a full GPU (a slice ran ~5x slower and crashed, see MESSY_LOOKUP.md)"
fi
gpu_mb="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
(( gpu_mb / 1024 >= MIN_GPU_GB )) || fail "$gpu_name has $((gpu_mb / 1024)) GB; Aya Expanse 32B needs an 80 GB card"
ok "$gpu_name, $((gpu_mb / 1024)) GB"

step "Disk"
free_gb="$(df -BG --output=avail "$WORKSPACE" | tail -1 | tr -dc '0-9')"
(( free_gb >= MIN_DISK_GB )) || fail "$WORKSPACE has ${free_gb} GB free; needs ${MIN_DISK_GB} GB (model ~65 GB, two environments, caches)"
ok "${free_gb} GB free in $WORKSPACE"

step "Caches under $WORKSPACE (kept across pod restarts)"
POD_ENV="$WORKSPACE/.pod_env"
if [[ ! -f "$POD_ENV" ]]; then
  cat > "$POD_ENV" <<EOF
export UV_INSTALL_DIR=$WORKSPACE/.uv/bin UV_PYTHON_INSTALL_DIR=$WORKSPACE/.uv/python
export UV_TOOL_DIR=$WORKSPACE/.uv/tools UV_TOOL_BIN_DIR=$WORKSPACE/.uv/bin
export UV_CACHE_DIR=$WORKSPACE/.uv-cache HF_HOME=$WORKSPACE/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH=$WORKSPACE/.uv/bin:\$PATH
EOF
fi
grep -q "source $POD_ENV" ~/.bashrc 2>/dev/null || echo "source $POD_ENV" >> ~/.bashrc
# shellcheck disable=SC1090
source "$POD_ENV"
ok "$POD_ENV (sourced from ~/.bashrc)"

step "uv and just"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v just >/dev/null || uv tool install rust-just
ok "$(uv --version), $(just --version)"

step "Repo: $REPO_DIR on $BRANCH"
cd "$REPO_DIR"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  printf '   ! local changes in the repo: not switching branch or pulling\n'
else
  git fetch -q origin
  git checkout -q "$BRANCH"
  git pull -q --ff-only origin "$BRANCH"
fi
ok "$(git log --oneline -1)"

step "Training environment (just setup + the sft extra)"
just setup
uv pip install --python .venv/bin/python -e ".[sft]"
.venv/bin/python -c "import torch, peft, trl, bitsandbytes; assert torch.cuda.is_available(); print('   ✓ torch', torch.__version__, '| peft', peft.__version__, '| trl', trl.__version__)"

if [[ "$SKIP_VLLM" != "1" ]]; then
  step "vLLM (its own environment: $VLLM_ENV)"
  [[ -x "$VLLM_ENV/bin/python" ]] || uv venv -q --python 3.12 "$VLLM_ENV"
  "$VLLM_ENV/bin/python" -c "import vllm" 2>/dev/null || uv pip install --python "$VLLM_ENV/bin/python" vllm
  ok "vllm $("$VLLM_ENV/bin/python" -c 'import vllm; print(vllm.__version__)')"
fi

step "Hugging Face: token and access to $MODEL"
[[ -n "${HF_TOKEN:-}" ]] || fail "HF_TOKEN isn't set: add it as a RunPod secret (a token that has accepted $MODEL's licence)"
.venv/bin/python - "$MODEL" "$DATASET" <<'PY'
import sys
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

model, dataset = sys.argv[1:]
print("   ✓ logged in as", HfApi().whoami()["name"])
try:
    hf_hub_download(model, "config.json")
except GatedRepoError:
    sys.exit(f"⛔ no access to {model}: accept its licence at https://huggingface.co/{model} with this token's account")
try:
    HfApi().dataset_info(dataset)
except RepositoryNotFoundError:
    sys.exit(f"⛔ can't see {dataset}: is this token's account in the reagent-lexsi org?")
print(f"   ✓ access to {model} and {dataset}")
PY

if [[ "$SKIP_MODEL" != "1" ]]; then
  step "Model: $MODEL (~65 GB, into $HF_HOME; resumes if interrupted)"
  uv pip install --python .venv/bin/python -q hf_transfer
  .venv/bin/python - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download

path = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
print("   ✓", path)
PY
fi

step "Golden trajectories $DATASET_REV → data/golden/hub-$DATASET_REV"
.venv/bin/python - "$DATASET" "$DATASET_REV" "data/golden/hub-$DATASET_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, rev, out = sys.argv[1:]
print("   ✓", snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=out))
PY

step "Training examples (just train-data: healable to train on, all for the held-out check)"
just train-data --dataset "data/golden/hub-$DATASET_REV" --force
just train-data --dataset "data/golden/hub-$DATASET_REV" --select all --force

step "Done"
cat <<EOF
   Training environment: $REPO_DIR/.venv   (source .venv/bin/activate)
   vLLM:                 $VLLM_ENV/bin/vllm
   Model cache:          $HF_HOME
   Training examples:    $REPO_DIR/data/train/sft-healable.jsonl, sft-all.jsonl
   Next (SEL-65):        just sft-train --dry-run, then just sft-train, then just sft-serve
EOF
