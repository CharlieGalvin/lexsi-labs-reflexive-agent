# GRPO warm-up: Customer Lookup (SEL-34)

**Goal:** prove the training pipeline (gated model → GRPO with tool calls → LoRA adapter → eval)
on one 24 GB GPU, and show GRPO measurably improving tool use, before we train on τ²-bench.

**Why a toy task:** τ²-bench episodes are long and need an LLM-simulated customer and API keys.
Customer Lookup keeps the failure we care about, a hallucinated tool argument, small enough
to iterate on in minutes.

## The task

`src/reflexive/toy_lookup.py`: 200 fake customers (fixed seed) behind two tools,
`find_customer(name)` and `get_customer(customer_id)`. Each question asks for one field
(balance, data left, plan). The values are random, so the model can't guess them.

- **L1**, one call: "What is the account balance for customer C1023?"
- **L2**, two calls: "What is the plan name for Priya Shah?"
- **Splits by customer:** 320 train questions (160 customers), 80 eval questions
  (40 customers who never appear in training).

**Reward** (weights in the config): `correct` 1.0 (the starred answer matches) +
`tool_ok` 0.5 (it fetched the asked-about customer's record, and no call errored, used a
made-up ID or a real ID of the wrong customer) + `format` 0.2 (exactly one `*answer*`).
Maximum 1.7.

**Baseline lessons (5 Nov).** AgentTune's default tool-result wording ("use this result to
answer") stopped every rollout after one call, so `toy_lookup.HARNESS` replaces it in both
training and eval. And Tiny Aya copied the example ID from the tool description (a real
customer) into every name question, answering from the wrong customer's record. So the tool
descriptions carry no example IDs or names, and `wrong_customer` counts this failure.

## How GRPO learns, in pandas terms

For each question the model writes `num_generations` answers. Each answer's advantage is its
reward relative to its group:

```python
df["advantage"] = df.groupby("qid")["reward"].transform(lambda r: (r - r.mean()) / (r.std() + 1e-4))
```

So **a group where every answer gets the same reward teaches nothing**. The task must be
"sometimes right" for the base model (roughly 20–70%). And **the loss curve isn't meaningful**:
watch the reward.

## Run ladder (each must pass before the next)

| # | Command | Time | Proves |
|---|---|---|---|
| 0 | Load Tiny Aya and generate one reply (see the setup notes) | 15 min | GPU, HF token, gated licence |
| 1 | `just train-smoke` | ~10 min | **SEL-34 green run**: GRPO with tools runs and saves an adapter |
| 2 | `just eval-lookup` | ~10 min | Base-model success rate. Outside 10–85%? Change `--l2-share` (and the config's `l2_share`) |
| 3a | `just train configs/train/lookup_grpo.yaml --max-steps 5 --output-dir runs/lookup-try` | ~5 min | Config works, and gives seconds per step |
| 3 | `just train configs/train/lookup_grpo.yaml` | ~1–2 h | Reward rises during training |
| 4 | `just eval-lookup --adapter runs/lookup-grpo` | ~10 min | **The result:** held-out improvement over run 2 |
| 3b | `just train configs/train/lookup_grpo_vllm.yaml` (optional) | ? | vLLM speed-up, same reward curve? |

Add `--report-to wandb` to a training command for live charts, and
`--push reagent-lexsi/<name>` to upload the adapter and logs to a private Hub repo.

**Success:** runs 1 and 3 finish; reward trends up; eval `correct` rises by about 15 points or
more and `hallucinated_ids` falls; peak GPU memory and seconds per step are recorded (they set
the budget for the τ²-bench heal runs).

## What each run writes (`runs/<name>/`, git-ignored)

| File | What it is |
|---|---|
| `adapter_model.safetensors`, `adapter_config.json` | The LoRA adapter |
| `rewards.jsonl` | One row per scored rollout: step, question, each reward component, final reply |
| `trajectories.jsonl` | AgentTune's full rollouts, with tool calls (reward column empty) |
| `reward_inputs_debug.json` | What the reward function received on its first call. **Check it after run 3a.** |
| `run_summary.json` | Wall time, seconds per step, peak GPU memory |
| `eval_eval.csv` / `.json` | From `eval_lookup.py`: one row per question, plus the means |

```python
import pandas as pd
r = pd.read_json("runs/lookup-grpo/rewards.jsonl", lines=True)
r.groupby("call")[["reward", "correct", "tool_ok", "hallucinated_ids"]].mean().rolling(10).mean().plot()
r.groupby(["call", "qid"])["reward"].std().eq(0).groupby("call").mean().plot()  # share of groups that teach nothing
```

## What to watch during run 3

- **Reward** (`reward` in the logs): should rise. Flat after ~50 steps → `learning_rate: 3.0e-5`.
- **Groups with zero spread** (`frac_reward_zero_std` in TRL's logs): near 1 → nothing is learned.
- **Completion length**: a sudden jump or collapse usually means the model found a loophole.
- **GPU memory** (`watch -n 2 nvidia-smi`; on a MIG slice some columns show N/A, which is normal).
  Out of memory → lower `max_completion_length` or `per_device_train_batch_size` (keep
  `per_device_train_batch_size × gradient_accumulation_steps` divisible by `num_generations`).

## Known unknowns

1. **Reward inputs.** AgentTune passes the rollout's `conversations` and `responses` to reward
   functions; `reward_inputs_debug.json` confirms it. If `conversation_0` is null there,
   `tool_ok` is always 0 and needs fixing before run 3.
2. **Tiny Aya has no tool-calling chat template.** AgentTune falls back to describing the tools
   in the system prompt (it logs a warning). If run 2 shows `n_tool_calls` near 0, the calls
   aren't being parsed.
3. **Fallback model:** if Tiny Aya's base success is near 0, repeat runs 2–4 with
   `--model Qwen/Qwen2.5-0.5B-Instruct` (and `model:` in the config): it has native tool
   calling, isn't gated and is faster. That separates "pipeline broken" from "model can't do it".
4. **Rollouts run one at a time** in AgentTune's GRPO path, so steps are slower than the batch
   size suggests. Run 3a's seconds per step tells you the real cost.
