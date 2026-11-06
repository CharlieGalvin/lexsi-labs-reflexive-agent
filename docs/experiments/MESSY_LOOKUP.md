# Experiment A: Messy-ID Lookup, a full GRPO run with W&B

**Goal:** a complete GRPO training run on a task Tiny Aya can partly do, to learn what the
training metrics and learning curves look like, and to show training measurably reduces
wrong tool arguments.

**Why this format:** in a first test, Tiny Aya made a single tool call well (~79% correct) but
struggled with multiple tool calls: it never chained a name lookup into a record lookup, and
invented a customer ID instead. So we chose a one-tool-call format to trial the training
pipeline and GRPO on a task the model can partly do. Every question needs **one** call to
`get_customer`; what varies is how hard it is to get the **argument** right, the toy version
of τ²-bench's Arg hallucination.

## The task

`src/reflexive/messy_lookup.py`: 200 fake customers (fixed seed) behind one tool. Each
question asks for one field (balance, data left, plan); the values are random, so the model
can't guess them. The tool only accepts IDs in the standard form (`C` + 4 digits), so the
model must normalise them. After an error it may retry once with a corrected ID.

| Question type (`level`) | Example | What the model must do |
|---|---|---|
| `clean` | "…for customer C1004?" | Copy the ID |
| `lower_dash` | "…for customer c-1014?" | Drop the dash (case doesn't matter) |
| `spaced` | "…for customer C 1032?" | Drop the space |
| `number_only` | "…for account number 1009?" | Add the `C` |
| `spelled` | "…for customer C one zero zero four?" | Turn words into digits |
| `distractor` | "Hi, I'm customer C1031. What is … for my partner, customer C1004?" | Pick the right one of two real IDs |

Train: 480 questions (160 customers × 3). Eval: 120 questions on 40 customers never seen in
training.

**Reward** (weights in the config): `correct` 1.0 (the starred answer matches) + `tool_ok`
0.5 (it fetched the asked-about customer's record, and no call errored or used the wrong
customer) + `format` 0.2 (exactly one `*answer*`). Maximum 1.7. A rollout that gets the ID
wrong, then retries and answers correctly, scores `correct` but not `tool_ok`.

**Before running, write down your predictions** (which types does the base model fail? which
will training fix?). Comparing them with the results is the fastest way to build intuition.

## Run ladder

This is the project's **starter task**: Arm A (the untuned student) against one GRPO run.
All commands are **terminal**, run on the pod in `/workspace/lexsi-labs-reflexive-agent`.

| # | Command | Time | Writes / check |
|---|---|---|---|
| 1 | `just baseline` | ~10 min | **Arm A**: `results/base.metrics.json` + the `base` row in [EXPERIMENTS.md](../EXPERIMENTS.md). Overall `correct` ideally 20–70% |
| 2 | `just after messy-try --max-steps 3 --report-to none` | ~5 min | A trial: it runs; `runs/messy-try/run_summary.json` gives s/step and peak memory |
| 3 | `just after` | ~1–2 h | **The GRPO run** → `runs/messy-grpo/`. Watch W&B (guide below) |
| 4 | `just evaluate` | ~10 min | `results/messy-grpo.metrics.json` + a new EXPERIMENTS.md row |
| 5 | `just compare` | instant | The base-vs-GRPO table to post on the ticket |

Then commit `results/*.metrics.json` and `docs/EXPERIMENTS.md`. Metrics files are never
overwritten: a second GRPO run gets its own name (`just after messy-grpo-2`, then
`just evaluate messy-grpo-2` and `just compare base messy-grpo-2`). An eval that already ran
can be recorded without re-running it: `just record runs/base-messy/eval_eval.csv base --arm A`.

**Metrics** (all on the 120 eval questions, greedy decoding): `task_success` = share answered
correctly, with a 95% interval; `calls/success` = tool calls ÷ correct answers;
`halluc. rate` = share of questions with an Arg error (an ID that doesn't exist, or a real ID
of the wrong customer).

**Tuning the mix** (`task_options.styles` in the config): a type at ~100% teaches nothing
(lower its weight); a type at 0% may be out of reach (keep it, but don't make it most of the
data). Eval and training must use the same mix, so a change means a new baseline under a new
run name. To keep the adapter on the Hub (private), add
`--push reagent-lexsi/tiny-aya-messy-grpo-v0` to the step 3 command.

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

## Results (6 Nov 2026)

**Status:** pipeline proven (the starter-task ticket); GRPO did **not** measurably improve the
held-out score yet. One open question below.

Base = Arm A, the untuned Tiny Aya. Trained = 100 GRPO steps, LoRA, `configs/train/messy_grpo.yaml`.
Both on the same 120 held-out eval questions with greedy decoding. Files:
[base](../../results/base.metrics.json), [messy-grpo](../../results/messy-grpo.metrics.json),
[comparison](../../results/base-vs-messy-grpo.compare.md); the W&B run is linked in the
`messy-grpo` row of [EXPERIMENTS.md](../EXPERIMENTS.md).

| | Base | Trained | Change |
|---|---:|---:|---:|
| **task_success** | 64/120 = 53.3% (95% 44.4–62.0%) | 67/120 = 55.8% (95% 46.9–64.4%) | +2.5 pts (within noise) |
| tool_ok (right record, no bad call) | 61% | 61% | 0 |
| halluc. rate (questions with an Arg error) | 39.2% (47/120) | 39.2% (47/120) | 0 |
| calls / success | 1.88 | 1.84 | −0.04 |
| `clean` | 19/22 | 19/22 | 0 |
| `spaced` | 18/21 | 19/21 | +1 |
| `distractor` | 17/19 | 18/19 | +1 |
| `spelled` | 10/15 | 9/15 | −1 |
| `lower_dash` | 0/21 | 2/21 | +2 |
| `number_only` | 0/22 | 0/22 | 0 |

**Training run:** 100 steps in 54 min (32.2 s/step), 10.1 GB peak GPU memory.
[W&B run](https://forge.coreweave.com/wandb/reagent_wandb/lexsi-track3/runs/chkqk6eg).

**During training** (sampled at temperature 0.8, training customers; `rewards.jsonl`), the
share answered correctly rose clearly, most of all on `lower_dash`:

| Type | Steps 0–15 | 16–30 | 31+ | Rollouts (31+) |
|---|---:|---:|---:|---:|
| `clean` | 0.42 | 0.44 | 0.93 | 68 |
| `spaced` | 0.38 | 0.46 | 0.84 | 88 |
| `distractor` | 0.55 | 0.50 | 0.68 | 88 |
| `lower_dash` | 0.75 (4 rollouts) | 0.25 | **0.79** | 92 |
| `spelled` | 0.12 | 0.17 | 0.30 | 124 |
| `number_only` | 0.00 | 0.00 | 0.10 | 92 |

## Findings

1. **The pipeline works end to end:** Arm A baseline → GRPO with tool calls on Tiny Aya →
   LoRA adapter → held-out eval → metrics file and experiment-log row → comparison table.
2. **Fix before trusting any Tiny Aya GRPO run: stop tokens.** AgentTune's training rollouts
   stopped only at the tokenizer's end-of-text token, but Tiny Aya ends its turn with another
   (ids `[3, 6]`). Every turn ran to 256 tokens of junk, which was then trained on: 315 s/step
   and 13 GB. `train_grpo.py` now passes the model's own stop tokens: 59 s/step and 8.5 GB.
   The heal runs on τ²-bench need the same fix.
3. **The training charts were healthy.** `frac_reward_zero_std` was mostly 0 (most groups had
   a spread of rewards to learn from); `kl` stayed ~0 until step ~18 (the LoRA adapter starts
   at zero), then rose steadily to ~0.1 by step 40; completion length was stable at 40–50
   tokens (no rambling, nothing cut off); `reward_parts/correct` rose from ~0.2 to ~0.6.
4. **The total reward chart is mostly question-mix noise.** Each step trains on only 2
   questions, so a step that draws `lower_dash` or `number_only` looks bad and one with two
   `clean` questions looks good. Reading `rewards.jsonl` **per question type** showed the
   learning; the W&B reward line alone didn't.
5. **GRPO can only reinforce what the model sometimes tries.** `lower_dash` went from 0%
   (greedy) to 25% then 79% (sampled) as GRPO amplified the rollouts that happened to drop the
   dash. `number_only` stayed at 0 until the model started adding the "C" by chance (10% late
   in the run). A behaviour that is never sampled can't be learned.
6. **But the training gains barely reached the held-out greedy eval** (+2.5 pts overall;
   `lower_dash` 0 → 2/21 against 79% in training), and the hallucination rate didn't move.

## Open question: why didn't `lower_dash` transfer?

79% of sampled training rollouts dropped the dash, yet greedy decoding on held-out customers
did so 2 times in 21. The candidates, with the check for each:

| Explanation | Check | Status |
|---|---|---|
| It memorised training customers, not the skill (unlikely: 160 customers, a generic skill) | Greedy eval on training customers: `just eval-lookup --adapter runs/messy-grpo --split train` | to do |
| Greedy decoding behaves differently from sampling | Sampled eval on held-out customers: `just eval-lookup --adapter runs/messy-grpo --temperature 0.8` | to do |
| Training and eval render the prompt differently (AgentTune has two code paths: the trainer's and the standalone engine's) | If the sampled eval is also ~10%, compare the two rendered prompts | to do |

## Next steps

- Answer the open question before training again: a run that can't transfer to the eval
  can't show improvement, whatever its training curves say.
- Then, for a clearer learning signal: more questions per step (less mix noise), more steps,
  and more weight on the hard types (`number_only`, `lower_dash`, `spelled`), with a new
  baseline under a new run name since the mix changes.
