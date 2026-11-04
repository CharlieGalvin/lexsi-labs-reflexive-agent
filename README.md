# Reflexive Agents

**A self-healing loop for mitigating agent hallucination in production workflows**

> The incredible LBG RAI team is embarking on an epic conquest to self-heal a hallucinating agent using reinforcement learning.

Lexsi Labs AI Safety & Alignment Hackathon (Nov 2026) · Track 3: Agentic Fine-tuning
Problem statements: *Self-healing agents*, *Efficient trajectories*, *Reliable tool calling*,
*Reward design and reward-hacking defences*.

---

## 1. Abstract

Current agentic systems suffer from **hallucination drift**: a model correctly identifies a
tool but generates hallucinated arguments, or enters an infinite loop of incorrect reasoning.
Most systems treat these as terminal failures that need a human to "reset" the prompt.

This project introduces a **Reflexive self-healing agent** that treats hallucinations as
repairable data points rather than terminal failures. Using the Lexsi Labs stack, we build a
multi-step customer-service agent with database write tools (on τ²-bench) with **AgentTune**, and a closed loop
in which the agent's `EventLog` is monitored by a **DECIDE**-based decision engine. When a
failure is detected (a ToolNotFound error, a loop, or a hallucinated argument), the system:

1. **Detect & Classify** — categorise the failure as **Tool**, **Arg** or **Loop** using a
   groundedness check.
2. **Synthesise** — use a teacher model (**Aya Expanse 32B**) to generate a corrected
   trajectory.
3. **Retrain & Gate** — pass corrected trajectories into the AgentTune RL factory
   (**GRPO**, **BCO**) to update the student (**Tiny Aya**).
4. **Verify** — use **AuditKIT** as a deployment gate, comparing the healed model's success
   rate against the baseline; the healed adapter is promoted only if it wins.

Our goal is to show that self-healing not only improves success rate but significantly
**reduces steps per task and redundant tool calls**, moving agents from best-effort execution
towards verified correctness through automated, gated weight updates.

```
             ┌──────────────────────── production / eval rollouts ───────────────────────┐
 task ──► Student agent (Tiny Aya + LoRA) ──► EventLog ──► DETECT & CLASSIFY (Tool/Arg/Loop)
                 ▲                                                   │ failures
                 │ promote if gate passes                            ▼
        VERIFY: AuditKIT gate ◄── RETRAIN: AgentTune GRPO/BCO ◄── SYNTHESISE: teacher
        (healed vs. baseline)        (student weights)            (Aya Expanse 32B) →
                                                                  verified golden trajectories
```

## 2. Experiment design — "the proof"

**Task suites** (decision 2 Nov: public benchmarks instead of a synthetic suite, so our numbers are
comparable with published results; see the [dataset options doc](https://claude.ai/artifact/F3PaRT61su9kdYb6LcWtxe)):

- **[τ²-bench](https://github.com/sierra-research/tau2-bench)** (MIT): retail, airline and telecom
  customer-service tasks with database write tools and an LLM-simulated user, scored by final database
  state. We heal on the train split and report all arms on the official test split.
  **Splits are frozen in [`configs/splits.json`](configs/splits.json)** (SEL-36): train 142, dev 36
  (15 retail, 6 airline, 15 telecom, carved from τ²-bench's train by task), test 100.
  **Rule: test task IDs are never used for healing, teacher fixes, DSPy or choosing anything;**
  **dev tasks are never trained on.** Code that builds training data calls
  `reflexive.splits.assert_not_test`.
- **[BFCL v3 multi-turn](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)** (Apache-2.0):
  state-scored multi-turn function calling. Generalisation test only, never trained on; its
  missing-parameter and missing-function sets map onto Arg and Tool hallucination.

Our published data contribution is the curated set of verified teacher fixes for τ²-bench train
failures, labelled Tool / Arg / Loop.

**Four arms** (same student model, same harness, same test split, same decoding):

| Arm | Name | What it is | Weight updates? |
|---|---|---|---|
| **A** | Baseline (naive agent) | Standard AgentTune rollout, no healing | No |
| **B** | **Reflexive agent (ours)** | Full Detect → Synthesise → Retrain → Verify loop | Yes |
| **C** | Prompt-only control | "Be careful not to hallucinate" system prompt | No |
| **D** | Optimised-prompt control | DSPy-optimised prompt on the dev/train split | No |

C and D separate the effect of *weight updates* from the effect of *better instructions*: if B
beats D, the gain is not something prompt engineering alone could get. A useful extra
ablation if time allows: **B′ = B without the teacher** (retrain on the student's own
successful rollouts), to isolate the teacher's contribution.

**Key metrics** (all on the held-out test split, with bootstrap confidence intervals):

| Metric | Definition | Owner |
|---|---|---|
| Success Rate (SR) | verified outcome correct (τ²-bench evaluator / AuditKIT outcome oracle) | Curator |
| Hallucination Rate | episodes with a Tool or Arg failure, or an ungrounded final claim | Architect (classifier) |
| Loop Rate | episodes classified Loop | Architect, cross-checked with AuditKIT `agent_loop_detection` |
| Steps / redundant calls | tool calls per task, repeated identical calls | Curator (AuditKIT `redundant_tool_calls`) |
| TTC | tokens and wall-clock time per completed task | Trainer (rollout logs) |
| Healing Accuracy | share of failures that the healed model gets right on re-run | Trainer (heal loop logs) |
| Success per token / ROI | SR ÷ tokens per task | Curator (final charts) |

## 3. Technical roadmap and roles

### Role 1 — Data & Synthesis ("the Curator")
**Tooling:** CuratorKIT, AuditKIT
- Build the task suite and the **"hard hallucination" dataset** (tasks that reliably trigger
  Tool / Arg / Loop failures in the baseline).
- Run the teacher (**Aya Expanse 32B**) to generate **golden trajectories** for failed tasks;
  verify each one by executing it against the sandbox before it is used for training.
- Run the AuditKIT verification suite; produce the final ROI and success-per-token charts.

### Role 2 — Systems & Logic ("the Architect")
**Tooling:** AgentTune, DECIDE
- Build the `ReflexiveAgent` class.
- Implement `classify_hallucination` (Tool vs. Arg vs. Loop).
- Integrate the DECIDE YAML workflow to orchestrate Detect → Heal, and build the public demo
  (live UI or interactive notebook).

### Role 3 — Model & Training ("the Trainer")
**Tooling:** Tiny Aya, AgentTune RL factory
- Set up the GRPO / BCO training loop; manage student weights and the adapter lifecycle.
- Execute the HEAL training steps (retrain the student on the Curator's dataset).

Task breakdown, milestones and owners: [`docs/WORKPLAN.md`](docs/WORKPLAN.md).

---

## Mapping the proposal onto the stack

AgentTune already ships most of the closed loop under `agenttune.decide.closed_loop`, so the
work is mainly wiring plus our own classifier, rewards and evaluation. Who builds each part:

| Proposal step | AgentTune / AuditKIT component | Owner |
|---|---|---|
| Monitor EventLog | `agenttune.agentic.EventLog`, `FailureDetector` (scans DECIDE audit log) | Architect |
| Detect & Classify | `FailureClassifier` (LLM, 5 root causes) | Architect: our `classify_hallucination` (Tool / Arg / Loop) |
| Synthesise | `TrainingExampleGenerator` (chosen/rejected pairs), `create_distill_trainer` | Curator |
| Retrain | `RetrainingTrigger`, `create_agentic_trainer(algorithm="grpo"/"bco")` | Trainer |
| Gate | `DeploymentGate`, `ReplayValidator` | Curator (AuditKIT scorers) |
| Orchestrate | DECIDE YAML → `langgraph` (`GraphRunner`), `FullClosedLoop` | Architect |

## Stack

| Piece | Used for |
|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | Rollouts, GRPO/BCO, DECIDE, closed-loop components |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Gate + independent scorers: `redundant_tool_calls`, `agent_loop_detection`, `tool_call_validity`, `tool_selection` |
| [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | Hard-hallucination dataset, quality-gated golden trajectories |
| `CohereLabs/tiny-aya-global` | Student (gated: accept the licence on the Hub) |
| Aya Expanse 32B (Cohere Labs) | Teacher |
| Hugging Face Hub ([reagent-lexsi](https://huggingface.co/reagent-lexsi)) | Adapters, datasets, logs |
| W&B or Trackio | Training curves |

## Quick start

Needs two small tools, once per machine: [uv](https://docs.astral.sh/uv/) (Python + packages)
and [just](https://just.systems) (task runner). On a Mac: `brew install uv just`.
Don't `pip install just`: that's an unrelated Python package with the same name. Without
Homebrew, use `uv tool install rust-just`. Check with `which just`: it should not be inside `.venv`.

```bash
git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent && cd lexsi-labs-reflexive-agent
just setup-dev                  # .venv (Python 3.12) + τ²-bench with its data + test tools; creates .env
#   → open .env and add OPENAI_API_KEY (needed to score retail/airline) and/or OPENROUTER_API_KEY
#     (+ COHERE_API_KEY for the teacher: docs/TEACHER_SETUP.md)
just check                      # lint + tests
just tau2-first-run telecom     # 2 τ²-bench tasks: official score + cost (a few cents; retail/airline need OPENAI_API_KEY)
source .venv/bin/activate && hf auth login   # then accept the Tiny Aya licence on its model page

# GPU box
just setup                      # also installs the training libraries and AgentTune
```

Run `just` on its own to list every command. Python must be **3.12 or 3.13** (τ²-bench rejects
3.14); `just setup-dev` creates a 3.12 environment for you.

## Repo layout

```
configs/                 one YAML per experiment / arm (copy, don't edit in place)
  splits.json            frozen τ²-bench train / dev / test task IDs (SEL-36; never regenerate)
src/reflexive/           Python package: each role adds its modules here
  outcome.py             pass/fail for any τ²-bench trajectory via τ²-bench's own scoring (Curator)
  episode.py             one τ²-bench conversation our code drives: tools, customer, context guard (Curator)
  llm.py                 API model names and OpenRouter prices
  splits.py              load the frozen splits; split_of(), assert_not_test()
scripts/tau2_*           τ²-bench: first run, cost report, context-fit check, OpenRouter pricing
third_party/             τ²-bench v2.0.0 (with data) and AgentTune, cloned by `just setup-dev` / `just setup`
tests/                   offline tests (no GPU) — run in CI on every PR
docs/TAU2_SETUP.md       τ²-bench setup decisions (splits, judges, context fit, costs)
docs/TEACHER_SETUP.md    teacher (Aya Expanse 32B) via the Cohere API: limits, tool calling, budget
docs/WORKPLAN.md         milestones + tasks per role → GitHub issues
docs/EXPERIMENTS.md      one row per run
results/*.metrics.json   committed evidence for the demo
```

## Team workflow

- Work is tracked in **Linear**: [Reflexive Agents project](https://linear.app/self-healing-hallucination/project/reflexive-agents-e8769e1623aa)
  (issues `SEL-5`…`SEL-30`, milestones M1–M4, role labels Curator / Architect / Trainer / Shared).
- Branch per issue using Linear's branch name (copy it from the issue, e.g. `charliegalvin/sel-12-…`),
  or mention `SEL-12` in the PR title — Linear moves the issue to In Progress / Done automatically.
- PR into `main`, CI must pass, one review.
- Every run gets a config file, an `EXPERIMENTS.md` row and a committed `metrics.json`.
- Models, adapters and datasets go to the team HF org [reagent-lexsi](https://huggingface.co/reagent-lexsi), not git.

## Licences

AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) are source-available for research and
non-commercial use, not OSI open source. Check the Tiny Aya and Aya Expanse licences before
publishing weights or teacher outputs. Code in this repo: pick a licence before going public.
