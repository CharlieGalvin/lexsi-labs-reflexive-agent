# Experiment A: Messy-ID Lookup, a full GRPO run with W&B

**Goal:** a complete GRPO training run on a task Tiny Aya can partly do, to learn what the
training metrics and learning curves look like, and to show training measurably reduces
wrong tool arguments.

**Why this task:** the Customer Lookup warm-up ([GRPO_WARMUP.md](GRPO_WARMUP.md)) showed Tiny
Aya makes one tool call well (~79%) but never chains two: given a name, it invented an ID
every time. So here every question needs **one** call to `get_customer`. What varies is how
hard it is to get the **argument** right, the toy version of τ²-bench's Arg hallucination.

## The task

`src/reflexive/messy_lookup.py`, on the same 200-customer database as the warm-up. The tool
only accepts IDs in the standard form (`C` + 4 digits), so the model must normalise them.

| Question type (`level`) | Example | What the model must do |
|---|---|---|
| `clean` | "…for customer C1004?" | Copy the ID |
| `lower_dash` | "…for customer c-1014?" | Drop the dash (case doesn't matter) |
| `spaced` | "…for customer C 1032?" | Drop the space |
| `number_only` | "…for account number 1009?" | Add the `C` |
| `spelled` | "…for customer C one zero zero four?" | Turn words into digits |
| `distractor` | "Hi, I'm customer C1031. What is … for my partner, customer C1004?" | Pick the right one of two real IDs |

Train: 480 questions (160 customers × 3). Eval: 120 questions on 40 customers never seen in
training. The reward is the same as the warm-up: `correct` 1.0 + `tool_ok` 0.5 + `format` 0.2.

**Before running, write down your predictions** (which types does the base model fail? which
will training fix?). Comparing them with the results is the fastest way to build intuition.

## Run ladder

All commands are **terminal**, run on the pod in `/workspace/lexsi-labs-reflexive-agent`.

| # | Command | Time | Check |
|---|---|---|---|
| 1 | `just eval-lookup --config configs/train/messy_grpo.yaml` | ~10 min | Per-type baseline. Overall `correct` ideally 20–70% (see below) |
| 2 | `just train configs/train/messy_grpo.yaml --max-steps 3 --output-dir runs/messy-try --report-to none` | ~5 min | It runs; `run_summary.json` gives s/step and peak memory |
| 3 | `just train configs/train/messy_grpo.yaml` | ~1–2 h | Watch W&B (guide below) |
| 4 | `just eval-lookup --config configs/train/messy_grpo.yaml --adapter runs/messy-grpo` | ~10 min | Per-type improvement over step 1 |

**After step 1, tune the mix** in `task_options.styles` of the config: a type at ~100% teaches
nothing (lower its weight); a type at 0% may be out of reach (keep it, but don't make it most
of the data). Re-run step 1 after any change: eval and training must use the same mix.
To keep the adapter on the Hub (private), add `--push reagent-lexsi/tiny-aya-messy-grpo-v0`
to the step 3 command.

## Reading the W&B charts

W&B groups the training charts under `train/`. The names below are what TRL logs (they can
differ slightly between versions). Each step only has 8 rollouts, so every curve is noisy: set
W&B's **smoothing** slider to about 0.8 and judge the trend, not single points.

| Chart | What it is | Healthy | Warning sign |
|---|---|---|---|
| `reward` | Mean total reward of the step's rollouts (max 1.7) | Noisy, trending up | Flat for 40+ steps: learning rate too low, or no signal (next row) |
| `reward_std` / `frac_reward_zero_std` | Spread of rewards inside each group / share of groups where all 4 answers scored the same | Spread > 0; zero-std share well below 1 | Zero-std share near 1: GRPO has nothing to learn from (task too easy or too hard) |
| `reward_parts/*` | Ours: batch mean of each reward part, plus `hallucinated_ids`, `wrong_customer` | `correct` and `tool_ok` up; `hallucinated_ids` down | `format` up while `correct` stays flat: it learned the stars, not the lookup |
| `kl` | How far the model has moved from the base model | Starts at 0, rises slowly | Shoots up: the model is drifting fast (a lower learning rate, or a higher `beta`, holds it back) |
| `entropy` | How uncertain the model's token choices are | Slowly falls as it becomes confident | Collapses to ~0: every answer in a group is the same, so no more learning |
| `completions/mean_length` (and `clipped_ratio`) | Tokens per rollout; share hitting `max_completion_length` | Stable or a little shorter | Jumps up or a high clipped ratio: rambling. Falls to almost nothing: degenerate answers |
| `loss` | The GRPO objective | Small and noisy; **not** a quality measure | Don't judge the run by it (unlike supervised training) |
| `grad_norm` | Size of each update before clipping (clipped at 1.0) | Mostly steady | Frequent large spikes: unstable updates |
| `learning_rate` | Fixed after 5 warm-up steps (`constant_with_warmup`) | Flat line | — |

**Check-ins during step 3:**
- **Step ~10:** charts are appearing, `kl` is near 0, and `frac_reward_zero_std` isn't ~1.
- **Step ~40:** the smoothed `reward` should be rising. If it's flat, stop the run (Ctrl+C) and
  try `learning_rate: 3.0e-5`.
- **The end:** `reward_parts/hallucinated_ids` lower than at the start. Look at the sample
  rollouts in W&B's completions table and at `rewards.jsonl` (below).

## Looking at the rollouts yourself (Python)

```python
import pandas as pd
r = pd.read_json("runs/messy-grpo/rewards.jsonl", lines=True)
if r["step"].isna().any():        # the training step, if TRL passed it; else count reward calls
    r["step"] = r["call"]
by_step = r.groupby("step")[["reward", "correct", "tool_ok", "hallucinated_ids"]].mean()
by_step.rolling(10).mean().plot(subplots=True, figsize=(8, 8))

# per question type: first 20 steps vs last 20
r["phase"] = pd.cut(r["step"], [-1, 20, 80, 10_000], labels=["early", "middle", "late"])
r.pivot_table(index="level", columns="phase", values="correct", observed=True).round(2)
```

## Setup on a new pod (terminal)

Use a full GPU, not a MIG slice: the slice ran at ~150 s/step and crashed with an
`NVML_SUCCESS == r INTERNAL ASSERT FAILED` error (most likely an out-of-memory error hidden by
MIG). Add `WANDB_API_KEY` (from wandb.ai/authorize) as a RunPod secret next to `HF_TOKEN`.

```bash
cat >> /workspace/.pod_env <<'EOF'
export UV_INSTALL_DIR=/workspace/.uv/bin UV_PYTHON_INSTALL_DIR=/workspace/.uv/python
export UV_TOOL_DIR=/workspace/.uv/tools UV_TOOL_BIN_DIR=/workspace/.uv/bin
export UV_CACHE_DIR=/workspace/.uv-cache HF_HOME=/workspace/hf_cache
export PATH=/workspace/.uv/bin:$PATH
EOF
echo 'source /workspace/.pod_env' >> ~/.bashrc && source /workspace/.pod_env
curl -LsSf https://astral.sh/uv/install.sh | sh && uv tool install rust-just
cd /workspace && git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent.git
cd lexsi-labs-reflexive-agent
git checkout SEL-34--Secure-a-GPU-and-get-just-setup-+-smoke-training-green-run
just setup
.venv/bin/python -c "import torch, wandb; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
[ -n "$WANDB_API_KEY" ] && echo "W&B key is set"
```

## Results

*(fill in after the run)*

| | Base | Trained | Change |
|---|---|---|---|
| Overall `correct` | | | |
| `hallucinated_ids` | | | |
| `wrong_customer` | | | |
| s/step, peak GPU memory | | | |
