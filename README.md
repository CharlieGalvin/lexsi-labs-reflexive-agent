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
2. **Synthesise** — use a teacher model (**GLM-5.3-Flash**, chosen in a bake-off) to
   generate a corrected trajectory.
3. **Retrain & Gate** — pass corrected trajectories into the AgentTune RL factory
   (**GRPO**, **BCO**) to update the student (**Aya Expanse 32B**).

*Model choices changed on 3 Nov:* the original proposal had Aya Expanse 32B as the teacher and
Tiny Aya as the student. Unaided Aya Expanse 32B solved 1 of 30 train tasks, so it became the
student (its 128K context also fits telecom), and GLM-5.3-Flash won the teacher bake-off. See
[`docs/experiments/`](docs/experiments/README.md).
4. **Verify** — use **AuditKIT** as a deployment gate, comparing the healed model's success
   rate against the baseline; the healed adapter is promoted only if it wins.

Our goal is to show that self-healing not only improves success rate but significantly
**reduces steps per task and redundant tool calls**, moving agents from best-effort execution
towards verified correctness through automated, gated weight updates.

```
             ┌──────────────────────── production / eval rollouts ───────────────────────┐
 task ──► Student (Aya Expanse 32B + LoRA) ─► EventLog ──► DETECT & CLASSIFY (Tool/Arg/Loop)
                 ▲                                                   │ failures
                 │ promote if gate passes                            ▼
        VERIFY: AuditKIT gate ◄── RETRAIN: AgentTune GRPO/BCO ◄── SYNTHESISE: teacher
        (healed vs. baseline)        (student weights)            (GLM-5.3-Flash) →
                                                                  verified golden trajectories
```

## 2. Experiment design — "the proof"

**Task suites** (decision 2 Nov: public benchmarks instead of a synthetic suite, so our numbers are
comparable with published results; setup, splits and costs: [`docs/TAU2_SETUP.md`](docs/TAU2_SETUP.md)):

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
- Freeze the τ²-bench train / dev / test splits, and build the **"hard hallucination"
  subset** (tasks that reliably trigger Tool / Arg / Loop failures in the baseline).
- Run the teacher (**GLM-5.3-Flash**) to generate **golden trajectories** on train tasks,
  verified by τ²-bench's own evaluator, then convert them to the student's format and publish
  them for the Trainer (**published 5 Nov: [v1](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories), private, 127 train + 31 dev reference
  records**; the hard subset follows as v1.1, SEL-60).
- Run the AuditKIT verification suite; produce the final ROI and success-per-token charts.

### Role 2 — Systems & Logic ("the Architect")
**Tooling:** AgentTune, DECIDE
- Build the `ReflexiveAgent` class.
- Implement `classify_hallucination` (Tool vs. Arg vs. Loop).
- Integrate the DECIDE YAML workflow to orchestrate Detect → Heal, and build the public demo
  (live UI or interactive notebook).

### Role 3 — Model & Training ("the Trainer")
**Tooling:** Aya Expanse 32B (student), AgentTune RL factory
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
| Aya Expanse 32B (Cohere Labs, CC-BY-NC) | Student |
| GLM-5.3-Flash (`z-ai/glm-5.3-flash`, MIT, via OpenRouter) | Teacher |
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
#   → open .env and add OPENAI_API_KEY (the retail/airline judges), OPENROUTER_API_KEY (teacher +
#     simulated customer), COHERE_API_KEY (the student, Aya Expanse 32B, for `just rollout`) and
#     HF_TOKEN (your own, for the team HF org): docs/TEACHER_SETUP.md
just check                      # lint + tests
just tau2-first-run telecom     # 2 τ²-bench tasks: official score + cost (a few cents; retail/airline need OPENAI_API_KEY)
source .venv/bin/activate && hf auth login   # then accept the Aya Expanse licence on its model page

# GPU box
just setup                      # also installs the training libraries and AgentTune
```

Run `just` on its own to list every command. Python must be **3.12 or 3.13** (τ²-bench rejects
3.14); `just setup-dev` creates a 3.12 environment for you.

## Repo layout

```
configs/                 one YAML per experiment / arm (copy, don't edit in place)
  splits.json            frozen τ²-bench train / dev / test task IDs (SEL-36; never regenerate)
  arms.yaml              the experiment arms: shared student and customer, one prompt per arm (SEL-41)
  prompts.yaml           the arms' versioned prompt texts (e.g. careful/v1); never edit a version in place
src/reflexive/           Python package: each role adds its modules here
  outcome.py             pass/fail for any τ²-bench trajectory via τ²-bench's own scoring (Curator)
  episode.py             one τ²-bench conversation our code drives: tools, customer, context guard (Curator)
  arms.py                the experiment arms (configs/arms.yaml) and one rollout of an arm (Architect)
  llm.py                 API model names and OpenRouter prices
  splits.py              load the frozen splits; split_of(), assert_not_test()
  teacher.py             Cohere `Action:` prompt format (to_prompted) and tool-call parsing; Cohere client
  golden.py              golden trajectories v1: agent's view, clean-pass pick, Cohere conversion, checks
scripts/rollout.py       run an arm on a domain's tasks: `just rollout A retail` (Arm A), `just rollout C ...`
scripts/tau2_*           τ²-bench: first run, cost report, context-fit check, OpenRouter pricing
scripts/teacher_*        teacher bake-off (teacher_sample) and the fix run (teacher_fixes)
scripts/*_report.py      bake-off and phase 2 statistics; student_review.py: Aya's failure types
scripts/make_splits.py   froze configs/splits.json (run once; --check verifies it)
scripts/build_golden.py  golden trajectories v1 → data/golden/v1/ (just golden)
scripts/publish_golden.py  private upload to the team HF org with a data card (just golden-publish)
third_party/             τ²-bench v2.0.0 (with data) and AgentTune, cloned by `just setup-dev` / `just setup`
tests/                   offline tests (no GPU) — run in CI on every PR
docs/DATA_CONTRACT.md    what the Curator hands over: the five rules, the files, record format and sizes
docs/TAU2_SETUP.md       τ²-bench setup decisions (splits, judges, context fit, costs)
docs/TEACHER_SETUP.md    teacher (GLM-5.3-Flash via OpenRouter) and the Cohere prompt format Aya uses
docs/experiments/        the teacher bake-off and golden trajectories, in order (start at its README)
docs/WORKPLAN.md         the original 2 Nov plan (live tracking is in Linear)
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
non-commercial use, not OSI open source. Check the Aya Expanse licence (CC-BY-NC) before
publishing weights, and the customer model's terms (GPT-4.1) before publishing trajectories. Code in this repo: pick a licence before going public.
