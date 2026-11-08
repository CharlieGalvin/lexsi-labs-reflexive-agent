# The indicative SFT run on a RunPod GPU, end to end (SEL-65)

QLoRA on Aya Expanse 32B, trained on the 112 healable golden fixes, then scored on the 36 dev
tasks next to Arm A. **Never run anything here on the test split.** This page is *how to run it*;
the question, the results, v1 against v2 and the decision rules are in [`SFT.md`](SFT.md).

Written from the first real run (7–8 Nov, H100 SXM): set-up about 45 minutes, training **23
minutes**, peak GPU memory **54 GB**, held-out loss **0.854 → 0.393**. GPU time for the whole
thing is about 2–3 hours (an H100 SXM is $3.99/hour; an A100 at $1.79 works too, just slower).

## Where things live on a pod

A RunPod pod has three kinds of space. Only the volume survives the pod stopping:

| Space | Path | Survives a stop? | Survives terminating the pod? | What goes there |
|---|---|---|---|---|
| **Network volume** (persistent) | `/workspace` | **Yes** | **Yes** (it's separate from the pod) | **Everything:** the repo, both environments, the model cache, the training data, the adapter, the dev results, the training output |
| Container disk (temporary) | `/`, `/root`, `/tmp` | No | No | Nothing that matters. It's wiped on every stop, so **tmux** and `~/.bashrc` go with it (see "After every restart") |
| RAM and GPU memory | running processes, tmux sessions | No | No | Only the model while it trains or serves: nothing to save |
| **Off the pod** (permanent) | Hugging Face, your Mac, git | Yes | Yes | The adapter (`--push`), the dev results (copied back), code (git) |

So: **work only under `/workspace`**. A *network* volume is tied to one data centre: you can only
get the GPUs that centre has free, but it outlives the pod, so the 65 GB model downloads once.
It costs about $0.07 per GB a month; delete it at the end of the project.

What `setup.sh` puts where, all under `/workspace`:

| What | Path | Size |
|---|---|---|
| Model cache (Aya Expanse 32B) | `/workspace/hf_cache` | ~65 GB |
| The repo, with its training environment (`.venv`) | `/workspace/lexsi-labs-reflexive-agent` | ~15 GB |
| vLLM's own environment | `/workspace/vllm-env` | ~10 GB |
| uv's Python and download cache | `/workspace/.uv`, `/workspace/.uv-cache` | ~5 GB + up to ~20 GB (clearable) |
| The adapter, its summary and step log | `<repo>/runs/sft-aya32b-r16/` | ~0.4 GB |
| Training output (everything printed) | `/workspace/sft-train.out` | small |

**The volume must be at least 150 GB.** At 100 GB the model download stopped with *"Disk quota
exceeded"* (37 GB in). `df` shows RunPod's whole shared storage, not your volume's limit, so the
setup's disk check can't catch it: check the size under **Storage** in the console. A volume can
be enlarged while in use. `uv cache clean` frees the package-download copies once installed.

## Before you start (once)

1. **Hugging Face:** accept the licence for
   [`CohereLabs/aya-expanse-32b`](https://huggingface.co/CohereLabs/aya-expanse-32b) with the
   account behind your token, which must be in the `reagent-lexsi` org. Use a **Write** token
   (`--push` uploads the adapter to the org). A token is shown only once when created; if you've
   lost it, check `~/.cache/huggingface/token` or the repo's `.env` on your Mac, or create a new one.
2. **Create the pod** (Pods → Deploy):
   - **GPU:** a full 80 GB card, not a MIG slice: H100 (SXM or PCIe) or A100. Bigger cards
     (H100 NVL, H200) don't help QLoRA.
   - **Template: PyTorch** (e.g. `runpod/pytorch:…-torch280-…`), **not** the vLLM one: the vLLM
     image starts a server on boot and pins torch for vLLM; `setup.sh` installs vLLM in its own
     environment beside the training one.
   - **Network volume** at `/workspace`, **150 GB or more**.
   - **Environment variables, before deploying:** `HF_TOKEN` and `OPENAI_API_KEY` (the customer
     and the judges in dev rollouts), as values or `{{ RUNPOD_SECRET_<name> }}`. Forgot? Pod → ⋮ →
     Edit Pod → Environment Variables; saving restarts the pod (only the container disk is lost).
   - Don't use "Subscribe" on an unavailable GPU: it deploys (and bills) whenever one frees up.

## Connecting

Use **SSH over exposed TCP** from the Connect tab (it supports `scp`; the port changes after every
restart, so copy it again), with keep-alives, **straight into a tmux session** (`-A` attaches to it
if it exists, or creates it):

```bash
ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -t root@<ip> -p <port> -i ~/.ssh/id_ed25519 "tmux new -A -s sft"
```

For a second job (e.g. the rollouts while the server runs), open a new Mac terminal tab and
connect the same way into another session, `"tmux new -A -s runs"`: no tmux shortcuts needed.

Check the keys arrived: `echo ${HF_TOKEN:+set} ${OPENAI_API_KEY:+set}` prints `set set`.

**Work inside tmux**, so a dropped connection or a closed laptop doesn't stop a long job:

| | |
|---|---|
| Start / reattach | `tmux new -s sft` / `tmux attach -t sft` |
| New window / switch | **Ctrl+b** together, release, then **c** / then **0**, **1**, … (or **n**, **p**) |
| Scroll | **Ctrl+b**, then **[**, arrows or Page Up, **q** to leave; or `tmux set -g mouse on` once |
| Check you're in it | `echo $TMUX` prints something |

The status bar at the bottom (`[sft] 0:bash*`) means you're in tmux.

## After every restart (the container disk was wiped)

```bash
apt-get update -qq && apt-get install -y -qq tmux
tmux new -s sft
source /workspace/.pod_env                       # the cache settings (lived in ~/.bashrc)
cd /workspace/lexsi-labs-reflexive-agent && source .venv/bin/activate && git pull
```

## 1. Setup (first time only; 30–45 min, mostly the download)

```bash
cd /workspace && git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent.git
cd lexsi-labs-reflexive-agent
git checkout charliegalvin/sel-66-train-data      # only until #49 merges: setup.sh isn't on main yet
bash scripts/pod/setup.sh
source .venv/bin/activate
```

Check out the branch **before** running the script: a clone starts on `main`, where the script
doesn't exist yet, and the script stays on whatever branch the repo is on (`BRANCH=` to switch).
It stops early on a MIG slice, a small card, or a token without access to the model, and it's safe
to re-run: it skips what's done (a re-run of a finished set-up takes a few minutes).

**If the model download fails with "Disk quota exceeded":** enlarge the volume (see above), run
`uv cache clean`, then resume the download (it keeps what's done):

```bash
.venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('CohereLabs/aya-expanse-32b', allow_patterns=['*.json','*.safetensors','*.model','*.txt','*.jinja'], max_workers=4))
" 2>&1 | tail -3
SKIP_VLLM=1 bash scripts/pod/setup.sh             # then finish the remaining steps
```

## 2. Dry run (2 min, no GPU)

```bash
just sft-train --dry-run
```

**Expect exactly:** `train: 112 conversations, 1,044,425 tokens (14% targets), longest 13,703;
held out: 15`. 14% targets is right: most of each conversation is context (the system prompt with
the tools, the customer, tool results); only the agent's ~150K tokens are learned. If the counts
are tiny (a few tokens a conversation) the script now stops with *"implausible tokenising"*.

## 3. Train (~25 min on an H100), with an automatic stop

```bash
(sleep 3h; runpodctl stop pod $RUNPOD_POD_ID) &
just sft-train --push 2>&1 | tee /workspace/sft-train.out; runpodctl stop pod $RUNPOD_POD_ID
```

- The pod **stops itself** when training ends, whether it succeeds or fails (the `;`), and after
  3 hours at the latest (the background backstop). Safe to go to bed once it's past step 3.
- `tee` keeps everything printed, errors included, in `/workspace/sft-train.out`: tmux's
  scrollback is lost when the pod stops.
- **Don't press Ctrl+C in that window:** it stops training, and the `;` then stops the pod.
- "Runpod config file not found" before `pod "…" stopped` is harmless: the stop works.

**What a good run looks like** (the 8 Nov run):

| | |
|---|---|
| Model load | ~30 s; `trainable params: 95,682,560 … trainable%: 0.2954` |
| Steps | 56 (2 passes of 28), ~24 s each; the held-out check every 7 steps (~20 s) |
| Training loss | jumps between steps (each step is a different batch of 4 conversations); ~0.7–0.9 at the start, ~0.28–0.36 by the end |
| Held-out loss | 0.854 before → 0.586 → 0.515 → … → 0.3935 → **0.3932**: falling, then flat (not rising: no memorising) |
| `grad_norm` | ~0.15–0.65, steady (worry if it reaches the tens or shows `nan`) |
| End | the adapter uploads (383 MB) to `reagent-lexsi/sft-aya32b-r16`, then `done in 23 min: …; peak 54.0 GB` |

The first 3 steps' losses can rise: the learning rate is still warming up (0 at step 1), so they
measure the untrained model on different batches.

**Checking the result without a GPU:** if `reagent-lexsi/sft-aya32b-r16` exists on Hugging Face,
training finished. For the numbers, deploy a **CPU** pod (cents an hour) with the same network
volume at `/workspace`, then:

```bash
cat /workspace/lexsi-labs-reflexive-agent/runs/sft-aya32b-r16/run_summary.json
tail -5 /workspace/lexsi-labs-reflexive-agent/runs/sft-aya32b-r16/log.jsonl
tail -30 /workspace/sft-train.out
```

## 4. Serve (5–15 min to start)

```bash
just sft-serve
```

In its own tmux window (or SSH tab). It first writes the chat template (below), then vLLM loads
the 16-bit model (slow from the network volume: 5–10 min) and prepares the GPU. Wait for
**"Application startup complete"**. It serves the base model with the adapter as `sft`, at a 32K
context. The GPU can't train and serve at once. There's no automatic stop from here: **stop the
pod yourself** when the dev runs are done.

## 5. Sanity check: the self-hosted base against Arm A (10 min)

In a second session (a new Mac tab: `"tmux new -A -s runs"`, then `source /workspace/.pod_env`,
`cd` and `source .venv/bin/activate` as above). Wait until the server answers:

```bash
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 15; done; echo READY
```

Then:

```bash
just rollout P retail --split dev --limit 5 --config configs/sft/arms.base.yaml
```

Arm A's student, on the pod instead of Cohere's API. It should look broadly like Arm A on the same
tasks (mostly failures, the odd pass, no crashes). Many crashes or empty replies mean the serving
doesn't match Cohere's: stop and check before step 6.

## 6. The real run: S (the adapter) and P (the control) on the 36 dev tasks (~1 hour each)

Both arms, in one go, then the pod stops itself (don't press Ctrl+C in this window). P is the fair
control: the same server, template and 32K context, without the adapter.

```bash
for a in S P; do
  cfg=$([ $a = S ] && echo configs/sft/arms.sft.yaml || echo configs/sft/arms.base.yaml)
  for d in retail airline telecom; do
    just rollout $a $d --split dev --config $cfg 2>&1 | tee -a /workspace/$a-dev.out
  done
done; runpodctl stop pod <pod id>
```

Progress, from another session: `grep -E "^\[[0-9]+/|^This run" /workspace/S-dev.out | tail -6`.
A re-run of the same command retries only crashed conversations. Score on the Mac after step 7:
`just oracle-score`, then `just arm-metrics` on each file, then `just arm-compare` P against S.

**Leave a credit buffer.** When the balance runs out, RunPod stops the pod mid-run: finished
conversations are safe on the volume, and the re-run picks up the rest. Plan for ~$3/hour for the
H100 plus the 5-minute model load on every start.

**Result (8 Nov, indicative; SEL-65):** S 16/36 (44%), P 5/36 (14%), Arm A 2/36 (6%). Paired by
task, 14 tasks passed only with the adapter and 3 only without (exact McNemar p ≈ 0.013); the
gain is in retail (2 → 8) and telecom (0 → 7), not airline (3 → 1 of 6).

## 7. Copy the results off the pod, then stop it

The results live on the network volume, so they're safe, but copy them to your Mac to work with
them. From the Mac (SSH over TCP supports `scp`; use the pod's current port):

```bash
cd ~/path/to/lexsi-labs-reflexive-agent
scp -P <port> -i ~/.ssh/id_ed25519 \
  "root@<ip>:/workspace/lexsi-labs-reflexive-agent/results/episodes/arms/{P,S}-*-dev.jsonl" results/episodes/arms/
scp -P <port> -i ~/.ssh/id_ed25519 \
  "root@<ip>:/workspace/{S,P}-dev.out" results/
```

Then **stop the pod** in the console. Terminating it is fine too: the network volume keeps
everything.

## What went wrong on the first run, and what fixed it

All fixed in the code now; listed so the symptoms are recognisable.

| Symptom | Cause | Fix |
|---|---|---|
| `setup.sh: No such file or directory` | The clone was on `main`; the script is on the branch | Check out the branch first (step 1) |
| `does not have an extra named sft` | The script checked out `main` mid-run (it defaulted to `main`) | It now stays on the repo's branch |
| `Disk quota exceeded` during the download | A 100 GB volume | 150 GB or more; `uv cache clean` |
| `Conversation roles must alternate` (dry run) | Aya's template refuses a conversation that starts with the agent's greeting, as every τ²-bench one does (Cohere's own API accepts it) | Training and serving use the template minus that one check (`reflexive.sft.relaxed_template`) |
| `224 tokens … longest 2` (dry run) | transformers 5 returns a `BatchEncoding` from `apply_chat_template`; its two keys were read as tokens | Read as token ids; implausible counts now stop the run |
| `unexpected keyword argument 'warmup_ratio'` | transformers 5 merged it into `warmup_steps` | Translated automatically |
| vLLM: `max_model_len (32768) is greater than the derived max_model_len (… 8192 …)` | Aya Expanse 32B's `config.json` says 8192; the model serves 128K (Cohere's API runs it at 128K) | `serve.sh` overrides `max_position_embeddings` (see the CUDA assert row) |
| vLLM: `Engine core initialization failed`, above it `FileNotFoundError: … 'ninja'` | FlashInfer compiles its sampling kernel on first use; the pod has no `ninja` on PATH | `serve.sh` sets `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch's sampler) |
| The server dies mid-run (`CUDA error: device-side assert triggered` in the log); every conversation after it is `BROKEN … Connection error` | `VLLM_ALLOW_LONG_MAX_MODEL_LEN` let conversations past config.json's 8,192 positions run off the end of vLLM's RoPE table | `serve.sh` passes `--hf-overrides '{"max_position_embeddings": MAX_LEN}'` instead |
| vLLM: `To serve at least one request with the model's max seq len (32768), (5.0 GiB KV cache is needed …) (4.88 GiB)` | At `--gpu-memory-utilization 0.92`, the 60 GiB of weights, activations and CUDA graphs leave 4.9 GiB for the KV cache (~160 KB a token) | `serve.sh` defaults to `GPU_UTIL=0.95`: ~7 GiB, ~45K tokens across all the conversations in flight |
| The SSH connection drops ("Operation timed out", "Broken pipe") and the server with it | Idle SSH timed out; the server ran outside tmux | `ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=8 -t … "tmux new -A -s sft"`: keep-alives, and straight into tmux |
| `duplicate session: sft` (and the next commands run outside tmux) | A session of that name already exists | `tmux attach -t sft` (or `tmux new -A -s sft`, which attaches or creates) |
| `CUDA out of memory … Tried to allocate 11.86 GiB` at step 2 | Logits over Aya's ~256K vocabulary at every position of a 13.7K-token conversation, plus their float32 copy | The loss scores only the positions before each target (`logits_to_keep`), in chunks: the same loss, ~20 GB less |
| A telecom conversation ends `BROKEN ContextWindowExceededError … maximum context length is 32768` and every re-run retries it | Only Cohere's wording ("too many tokens") was recognised as an overflow | `reflexive.teacher` also recognises vLLM's; it's scored as a fail with `context_overflow`, as for Arm A |
| The pod stops mid-run and a new one ("…-migration-2") appears after a top-up | The balance ran out | Start the new pod: the volume moves with it. Re-run the same commands; finished conversations are kept |
| The pod stopped after Ctrl+C | The `;` ran the stop after the interrupted training | Don't press Ctrl+C in the training window |

## Other problems

| Symptom | What to do |
|---|---|
| `setup.sh` stops at the GPU check | Recreate the pod on a full 80 GB card (not MIG) |
| "no access to CohereLabs/aya-expanse-32b" | Accept the licence with the token's account, then re-run `setup.sh` |
| Training still runs out of GPU memory | Lower `max_len` in `configs/train/sft_aya32b.yaml` to 14000 (the longest example is 13,703 tokens) |
| vLLM runs out of memory at start-up | `MAX_LEN=16384 just sft-serve`, and set `max_context: 16384` in both `configs/sft/` files to match |
| Rollouts can't connect | Is `sft-serve` still running, and did it print "Application startup complete"? |
| Many crashed conversations | Run the same command again; if the same tasks crash every time, keep the output for review |
| `du` prints nothing for minutes | It's walking tens of thousands of files on network storage; Ctrl+C, it isn't needed |
