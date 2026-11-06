# PPO, GRPO, DPO and BCO: Choosing the Training Algorithm for Reflexive Agents

5 November 2026

## Overview

All four methods nudge an already-capable model toward better behaviour; the key split is **online** (learn from fresh attempts scored by a reward) versus **offline** (learn from a fixed, labelled dataset).

- **PPO and GRPO** are online reinforcement learning: the model acts, gets scored, and updates. Think "learn by doing".
- **DPO and BCO** are offline preference learning: the model studies examples of good and bad answers. Think "learn from examples".

AgentTune exposes all four (plus RLOO) through one entry point, `create_agentic_trainer(algorithm=...)`, so switching between them is mostly a configuration change. The question for this project is which one fits our data, our reward and our compute budget.

Scope: this project targets the *self-healing agents* problem statement. Reward design and reward-hacking defences are out of scope, so we reuse existing verifiable rewards rather than designing new ones.

## PPO (Proximal Policy Optimization)

PPO is the classic online RL algorithm and the engine behind the original RLHF recipe, but it is the heaviest of the four.

The model generates responses and a **reward model** scores them. PPO also trains a **value model (critic)**, a second network that predicts how good a situation is, so it can tell whether an action was better or worse than expected. Each update is clipped (the "proximal" part) so the model cannot change too drastically in one step.

- **Strengths:** powerful, well studied, general purpose.
- **Weaknesses:** four models in memory (policy, reference copy, reward model, critic), and fiddly to tune.
- **Link to GRPO:** GRPO is essentially PPO without the critic.

## GRPO (Group Relative Policy Optimization)

GRPO learns directly from a reward function by comparing attempts against each other, with no critic and no reward model to train.

For each prompt the model generates a group of attempts (for example 8). A reward function scores each one, and every attempt is judged relative to its group: those above the group average are reinforced, those below are suppressed.

- **Needs:** a reward function that can run automatically, and the ability to generate and score rollouts during training.
- **Strengths:** lighter than PPO; can discover behaviour that is not in any dataset; optimises the outcome we actually care about.
- **Weaknesses:** expensive, because every training step needs fresh multi-turn rollouts; sensitive to reward design; prone to reward hacking; gives no signal when every attempt in a group scores the same.

## DPO (Direct Preference Optimization)

DPO skips reinforcement learning entirely and trains on pairs of a better and a worse answer to the same prompt.

Each example has a *chosen* and a *rejected* response. A simple loss makes the chosen response more likely than the rejected one, relative to a reference model. There is no reward model, no critic and no live generation, so it behaves more like supervised learning than RL.

- **Strengths:** simple, stable and cheap; the most widely used preference method.
- **Weaknesses:** needs genuinely paired data; offline, so it cannot explore beyond what is in the dataset.
- **Fit:** our pipeline naturally produces pairs (the student's failed attempt and the teacher's corrected one), which suits DPO's data format.

## BCO (Binary Classifier Optimization)

BCO is DPO's looser cousin: it learns from a plain thumbs-up or thumbs-down on each example, with no pairing required.

It treats the problem as classification on the model's implicit reward, anchored to a reference model. Examples labelled desirable or undesirable can come from different prompts, which makes it handy for messy real-world data. KTO is a close relative.

- **Strengths:** cheap, stable and reproducible; copes with unbalanced data (many failures, few fixes).
- **Weaknesses:** can only learn from what is in the dataset, so it is capped by label and teacher-trajectory quality; can overfit a small dataset.
- **Fit:** verified teacher trajectories are the desirable examples and the student's failed rollouts are the undesirable ones.

## Comparison

PPO and GRPO need a live reward; DPO and BCO need a labelled dataset; GRPO is the lightest of the online options.

| | PPO | GRPO | DPO | BCO |
| --- | --- | --- | --- | --- |
| Online or offline | Online | Online | Offline | Offline |
| Needs reward signal | Learned reward model | Reward function | Preference pairs | Binary labels |
| Extra models to train | Critic + reward model | None | None | None |
| Cost | Highest | High | Low | Low |
| Complexity | High | Medium | Low | Low |

## Why GRPO is the best fit for this project

GRPO is the only option that optimises the outcomes we are trying to improve (success, fewer steps, fewer redundant calls) directly, on the student's own behaviour, without the overhead of PPO.

1. **We already have verifiable rewards.** τ²-bench scores tasks by final database state, and AuditKIT provides scorers for `redundant_tool_calls`, `agent_loop_detection`, `tool_call_validity` and `tool_selection`. GRPO can optimise these directly, including the efficiency metrics that are our headline claim beyond success rate. We reuse them as they are rather than designing new rewards.
2. **It learns from the student's own failures.** Hallucination drift happens in the student's own distribution of tool calls and arguments. Online rollouts expose it to exactly those states; DPO and BCO only see whatever the dataset happened to contain.
3. **It is built for multi-turn tool calling.** AgentTune describes GRPO as the default for agentic training, and all its trainers run tool-calling rollouts rather than plain-text generation.
4. **It avoids PPO's overhead.** No critic and no reward model means less memory and fewer moving parts, which matters for a small student (Tiny Aya) with LoRA on limited GPU time.
5. **It fits the self-healing loop.** The loop retrains the student on its own failures and promotes the healed model only if it beats the baseline. GRPO trains on the same verified outcome signal that the AuditKIT gate measures, so what we optimise and what we verify are the same thing.

## How much data each method needs

There is no fixed number: GRPO needs few prompts but many rollouts, while DPO and BCO need a clean labelled dataset, and our 142 training tasks are enough for a pilot but thin for anything more.

The scales below are rough rules of thumb for a narrow task, not figures from the AgentTune documentation.

| Method | What counts as data | Rough scale |
| --- | --- | --- |
| GRPO | Prompts plus a reward function (no labelled answers) | Low hundreds to a few thousand prompts, each reused across epochs and sampled several times |
| PPO | Prompts plus a reward model | Same prompt scale as GRPO, plus the reward model's own comparison data unless the reward is rule-based |
| DPO | Chosen/rejected pairs | A few hundred can shift behaviour on a narrow task; a few thousand is typical |
| BCO | Individually labelled good/bad examples | Similar to DPO; tolerates unbalanced labels |

**What we actually have** (from `docs/TAU2_SETUP.md`, split frozen 4 Nov):

| Domain | Train | Dev (never trained on) | Test |
| --- | --- | --- | --- |
| Retail | 59 | 15 | 40 |
| Airline | 24 | 6 | 20 |
| Telecom | 59 | 15 | 40 |
| Total | 142 | 36 | 100 |

If telecom is dropped under the context-window rule, training falls to 83 tasks and the test set to 60.

- **GRPO: only mixed-outcome tasks teach it anything.** A task the student always fails or always passes gives a flat group and no signal. After the Arm A runs, count the tasks that pass in some trials and fail in others; that count is the real GRPO dataset and may be a fraction of 142.
- **GRPO: cost is rollouts, not prompts.** Total conversations are roughly steps × prompts per step × generations per prompt. For example, 100 steps × 8 prompts × 4 generations is 3,200 conversations, each with a simulated customer. Multiply by the measured user cost per task to budget.
- **GRPO: airline is thin.** At 24 training tasks it is the likeliest domain to overfit.
- **GRPO: telecom's extra pool may be unusable.** It has a 2,285-task generated pool, but its prompt alone fills about 91% of Tiny Aya's 8,192-token window, so the pool only helps if that fit problem is solved.
- **DPO and BCO: capped by verified teacher fixes.** The dataset is the set of train failures that the teacher fixes and the sandbox verifies, probably tens to roughly 100 examples. Sampling several fixes per task enlarges it but risks memorising a few tasks, which is why they suit a warm start better than the whole method.
- **Trainer note.** AgentTune's own loop has no simulated customer, so GRPO on τ²-bench needs a `rollout_func` that drives `Tau2Episode`.

To settle the numbers empirically, run a short pilot and plot held-out dev success against the number of training tasks used; if the curve is still rising, more data will help.

## Risks and when to use BCO or DPO instead

GRPO is our primary method, but it carries real risks, and BCO and DPO are the cheap fallbacks.

- **Reward gaming.** A model can score well without solving the task, for example by skipping tool calls to avoid argument errors. We are not designing new rewards, so the safeguard is to use the existing τ²-bench outcome check unchanged and read a sample of completions during each run.
- **No learning signal.** If the student scores 0 on every attempt, or every attempt in a group scores the same, GRPO learns nothing. A warm start from teacher trajectories (BCO, DPO or distillation with `create_distill_trainer`) can fix this.
- **Cost.** Each GRPO step needs fresh multi-turn rollouts, and τ²-bench uses an LLM-simulated user, so rollouts cost API calls and GPU time. Keep early runs small and measure seconds per step before scaling.
- **Use BCO or DPO** when GPU or API budget runs short, or to get a first end-to-end loop working quickly on the Curator's verified trajectories.
- **Ablation worth running:** compare GRPO against BCO on the same data, so the report can show what online learning adds over learning from fixed examples.
