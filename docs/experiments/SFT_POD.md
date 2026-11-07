# The indicative SFT run on a RunPod H100, end to end (SEL-65)

QLoRA on Aya Expanse 32B, trained on the 112 healable golden fixes, then scored on the 36 dev
tasks next to Arm A. Allow **3–4 hours of pod time** (about $2–3 an hour on an H100), plus a few
dollars of GPT-4.1 for the customer and the judges. **Never run anything here on the test split.**

## Where things live on a pod

A RunPod pod has three kinds of space. Only one of them survives the pod stopping:

| Space | Path | Survives a stop? | Survives terminating the pod? | What goes there |
|---|---|---|---|---|
| **Volume disk** (persistent) | `/workspace` | **Yes** | No | **Everything:** the repo, both environments, the model cache, the training data, the adapter, the dev results |
| Container disk (temporary) | `/`, `/root`, `/tmp` | No | No | Nothing that matters. `setup.sh` points every cache at `/workspace` so nothing big lands here |
| RAM and GPU memory | `/dev/shm`, running processes, tmux sessions | No | No | Only the model while it trains or serves: nothing to save |
| **Off the pod** (permanent) | Hugging Face, your Mac, git | Yes | **Yes** | The adapter (`--push`), the dev results and summaries (copied back), code changes (git) |

So: **work only under `/workspace`**, and **before terminating the pod, push the adapter and copy
the results back**. "Stop" keeps `/workspace` (you pay a little for the disk), but the same GPU
may not be free when you restart. "Terminate" deletes `/workspace` too.

What `setup.sh` puts where, all under `/workspace`:

| What | Path | Size |
|---|---|---|
| The repo, with its training environment (`.venv`) | `/workspace/lexsi-labs-reflexive-agent` | ~15 GB |
| vLLM's own environment | `/workspace/vllm-env` | ~10 GB |
| Model cache (Aya Expanse 32B) | `/workspace/hf_cache` | ~65 GB |
| uv's caches and Python | `/workspace/.uv`, `/workspace/.uv-cache` | ~10 GB |
| Training examples | `<repo>/data/train/sft-{healable,all}.jsonl` | ~10 MB |
| The adapter and its summary | `<repo>/runs/sft-aya32b-r16/` | ~0.5 GB |
| Dev results | `<repo>/results/episodes/arms/{P,S}-*-dev.jsonl` | ~5 MB |

Hence the **200 GB volume**: about 100 GB used, plus room for checkpoints and a second run.

## Before you start (once)

1. **Hugging Face:** accept the licence for
   [`CohereLabs/aya-expanse-32b`](https://huggingface.co/CohereLabs/aya-expanse-32b) with the
   account behind your `HF_TOKEN`. The token's account must be in the `reagent-lexsi` org.
2. **RunPod secrets** (Settings → Secrets): `HF_TOKEN`, `OPENAI_API_KEY` (the customer and the
   judges in dev rollouts), optionally `WANDB_API_KEY` (training charts).
3. **Create the pod:** an **80 GB GPU, a full one (not a MIG slice)**: A100 or H100. Use the
   RunPod **PyTorch** template, not the vLLM one: `setup.sh` installs vLLM in its own environment
   beside the training one. Attach a volume of at least 150 GB at `/workspace`. **Before
   deploying,** add the environment variables `HF_TOKEN` (a *Write* token: `--push` uploads to the
   org) and `OPENAI_API_KEY`, as values or as `{{ RUNPOD_SECRET_<name> }}`. Forgot? Edit Pod →
   Environment Variables restarts the pod; only the temporary container disk is lost.

## On the pod

Connect with **SSH over exposed TCP** (the Connect tab; it also supports `scp`), check the keys
are there (`echo ${HF_TOKEN:+set} ${OPENAI_API_KEY:+set}` prints `set set`), and **work inside
tmux**, so a dropped connection doesn't stop a long job: `tmux new -s sft` (reattach with
`tmux attach -t sft`; scroll with `Ctrl-b [` then the arrow keys, `q` to leave, or
`tmux set -g mouse on` once to scroll with the trackpad). If tmux is missing:
`apt-get update -qq && apt-get install -y -qq tmux`.

### 1. Setup (30–45 min, mostly the download)

```bash
cd /workspace && git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent.git
cd lexsi-labs-reflexive-agent
git checkout charliegalvin/sel-66-train-data      # only until #49 merges: setup.sh isn't on main yet
bash scripts/pod/setup.sh
source .venv/bin/activate
```

The script stays on the branch the repo is on (set `BRANCH=` to switch). It stops early, before installing anything,
on a MIG slice, a card under 80 GB, too little disk, or a token without access to the model. It's
safe to re-run: it skips what's done. It ends by building the training examples.

### 2. Dry run (2 min, no GPU work)

```bash
just sft-train --dry-run
```

**Check:** `train: 112 conversations`, `held out: 15`, the longest about 12,700 tokens, and no
"left out" line. If it stops with **"rewrote earlier tokens"**, Aya's chat template doesn't fit
the masking: send the output to Charlie's Claude session before going further.

### 3. Train (~1 hour)

```bash
just sft-train --push
```

`--push` uploads the adapter to `reagent-lexsi/sft-aya32b-r16` (private) when training ends, so
it survives the pod. **Watch:** the training loss should fall; the held-out loss is printed
before and after. If the held-out loss **rises** a lot while the training loss falls, the adapter
is memorising: note it before using it. The run writes `runs/sft-aya32b-r16/run_summary.json`
(losses, minutes, peak GPU memory).

### 4. Serve (5 min to start)

```bash
just sft-serve
```

Leave it running (its own tmux window: `Ctrl-b c`). Wait for **"Application startup complete"**.
It serves the 16-bit base with the adapter as `sft`, at a 32K context. Training must be finished:
the GPU can't do both.

### 5. Sanity check: the self-hosted base against Arm A (10 min)

In a second tmux window (`Ctrl-b c`, then `source .venv/bin/activate`):

```bash
just rollout P retail --split dev --limit 5 --config configs/sft/arms.base.yaml
```

This is Arm A's student on the pod instead of Cohere's API. It should look broadly like Arm A on
the same tasks (mostly failures, the odd pass, no crashes). Many crashes or empty replies mean the
serving doesn't match Cohere's: stop and check before step 6.

### 6. The real run: the SFT adapter on the 36 dev tasks (~45 min)

```bash
just rollout S retail  --split dev --config configs/sft/arms.sft.yaml
just rollout S airline --split dev --config configs/sft/arms.sft.yaml
just rollout S telecom --split dev --config configs/sft/arms.sft.yaml
just oracle-score results/episodes/arms/S-*-dev.jsonl
just arm-metrics results/episodes/arms/S-retail-dev.jsonl --owner Charlie
just arm-metrics results/episodes/arms/S-airline-dev.jsonl --owner Charlie
just arm-metrics results/episodes/arms/S-telecom-dev.jsonl --owner Charlie
```

If a command reports crashed conversations, run it again (it retries only those). Compare with
**Arm A on dev: 2/36 (5.6%)**, and look at the Tool / Arg / Loop counts and how many
conversations hit the 32K context (`cut_off` in the metrics).

### 7. Copy the results off the pod, then stop it

The dev results and summaries live only on the pod's volume until you copy them. With RunPod's
`runpodctl` (installed on pods; on the Mac: `brew install runpod/runpodctl/runpodctl`):

```bash
# on the pod
tar czf /workspace/sft-results.tgz results/episodes/arms/P-*-dev.jsonl results/episodes/arms/S-*-dev.jsonl \
    results/arm-S-*.metrics.json runs/sft-aya32b-r16/run_summary.json runs/sft-aya32b-r16/log.jsonl
runpodctl send /workspace/sft-results.tgz        # prints a one-time code
# on the Mac, in the repo
runpodctl receive <code> && tar xzf sft-results.tgz
```

(Or `scp` over the pod's SSH connection.) Then **stop the pod** from the RunPod console. Terminate
it only once the adapter is on Hugging Face and the results are on your Mac.

## If something goes wrong

| Symptom | What to do |
|---|---|
| `setup.sh` stops at the GPU check | Recreate the pod on a full 80 GB card (not MIG) |
| "no access to CohereLabs/aya-expanse-32b" | Accept the licence on Hugging Face with the token's account, then re-run `setup.sh` |
| Training runs out of GPU memory | Lower `max_len` in `configs/train/sft_aya32b.yaml` to 14000 (every example is under ~12,700 tokens) |
| vLLM runs out of memory at start-up | `MAX_LEN=16384 just sft-serve`, and set `max_context: 16384` in both `configs/sft/` files to match |
| Rollouts can't connect | Is `sft-serve` still running, and did it print "Application startup complete"? |
| Many crashed conversations | Run the same command again; if the same tasks crash every time, keep the output for review |
| The pod restarted | Reattach (`tmux attach -t sft`), `cd /workspace/lexsi-labs-reflexive-agent && source .venv/bin/activate`, re-run the step: everything under `/workspace` is still there |
