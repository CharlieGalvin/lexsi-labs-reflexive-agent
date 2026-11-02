# Reflexive Agents

**A self-healing loop for mitigating agent hallucination in production workflows**

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
multi-step workflow (a financial / business data analyst) with **AgentTune**, and a closed loop
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

**Task suite.** 200 multi-step queries that need SQL database access plus API calls, e.g.
*"Find the user with the highest churn in the last 30 days and email the regional manager a
summary of their last 3 failed orders."* Every task has a programmatically verifiable outcome
(final DB/API state or answer), and side-effecting tools (email, etc.) are sandboxed mocks that
record what was sent. Split: dev/train for healing and prompt optimisation, held-out test for
all reported numbers.

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

| Metric | Definition | Where it's computed |
|---|---|---|
| Success Rate (SR) | verified outcome correct | `metrics.task_success` + AuditKIT outcome oracle |
| Hallucination Rate | episodes with a Tool or Arg failure, or an ungrounded final claim | `healing.classify_hallucination`, `metrics.hallucination_rate` |
| Loop Rate | episodes classified Loop | `healing.classify_hallucination`, AuditKIT `agent_loop_detection` |
| Steps / redundant calls | tool calls per task, repeated identical calls | `metrics.avg_tool_calls`, `redundant_call_rate`, AuditKIT `redundant_tool_calls` |
| TTC | tokens and wall-clock time per completed task | rollout logs |
| Healing Accuracy | share of failures that the healed model gets right on re-run | heal loop logs |
| Success per token / ROI | SR ÷ tokens per task | `compare.py` charts |

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
- Implement `classify_hallucination` (Tool vs. Arg vs. Loop) — **a tested first version is in
  [`src/reflexive/healing.py`](src/reflexive/healing.py)**.
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
work is mainly wiring and our own classifier, rewards and evaluation:

| Proposal step | AgentTune / AuditKIT component | This repo |
|---|---|---|
| Monitor EventLog | `agenttune.agentic.EventLog`, `FailureDetector` (scans DECIDE audit log) | `rollout.py`, `trajectory.py` |
| Detect & Classify | `FailureClassifier` (LLM, 5 root causes) | **`healing.classify_hallucination`** (deterministic Tool/Arg/Loop, maps to AgentTune's taxonomy via `TO_AGENTTUNE`) |
| Synthesise | `TrainingExampleGenerator` (chosen/rejected pairs), `create_distill_trainer` | teacher config (Role 1) |
| Retrain | `RetrainingTrigger`, `create_agentic_trainer(algorithm="grpo"/"bco")` | `scripts/train.py`, `rewards.py` |
| Gate | `DeploymentGate`, `ReplayValidator` | AuditKIT scorers in `scripts/evaluate.py`, `compare.py` |
| Orchestrate | DECIDE YAML → `langgraph` (`GraphRunner`), `FullClosedLoop` | Role 2 |

## Stack

| Piece | Used for |
|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | Rollouts, GRPO/BCO, DECIDE, closed-loop components |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Gate + independent scorers: `redundant_tool_calls`, `agent_loop_detection`, `tool_call_validity`, `tool_selection` |
| [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | Hard-hallucination dataset, quality-gated golden trajectories |
| `CohereLabs/tiny-aya-global` | Student (gated: accept the licence on the Hub) |
| Aya Expanse 32B (Cohere Labs) | Teacher |
| Hugging Face Hub | Adapters, datasets, logs |
| W&B or Trackio | Training curves |

## Quick start

```bash
git clone https://github.com/CharlieGalvin/reflexive-agents && cd reflexive-agents
python3.12 -m venv .venv && source .venv/bin/activate
cp .env.example .env            # fill HF_TOKEN etc.; then: set -a; source .env; set +a
hf auth login                   # and accept the Tiny Aya licence on its model page

make setup-dev && make test     # laptop: classifier, rewards, metrics, tests — no GPU
make setup                      # GPU box: also installs AgentTune from source
make smoke                      # 10 training steps, checks the plumbing
make baseline                   # Arm A rollouts on the held-out split
make train                      # GRPO with the reward stack
make after evaluate compare     # tuned rollouts → metrics → before/after table
```

The starter task is BioGRID SQL QA (AgentTune's own example) so the whole pipeline runs on
day one; Role 1 replaces it with the SQL + API task suite above.

## Repo layout

```
configs/                 one YAML per experiment / arm (copy, don't edit in place)
src/reflexive/
  healing.py             Detect & Classify: Tool / Arg / Loop groundedness classifier
  trajectory.py          parses trajectories (OpenAI tool_calls AND Cohere/AgentTune JSON fallback)
  rewards.py             correctness · efficiency · groundedness · tool validity
  data.py                starter task + unanswerable probes
  tools.py               the agent's tools (read-only SQL to start)
  rollout.py             eval-time harness, identical for every arm
  metrics.py             headline metrics
scripts/                 train · collect · evaluate · compare · setup_github_project.sh
tests/                   offline tests (no GPU) — run in CI on every PR
docs/WORKPLAN.md         milestones + tasks per role → GitHub issues
docs/EXPERIMENTS.md      one row per run
results/*.metrics.json   committed evidence for the demo
```

## Reward design (and reward-hacking defences)

| Reward | Pays for | Defence |
|---|---|---|
| `correctness` | right final answer, or abstaining when the task is unanswerable | — |
| `efficiency` | few calls, no repeats, no failed calls, within step budget | **paid only if the answer is correct *and* backed by ≥1 successful tool result** — otherwise "answer without calling anything" is optimal |
| `groundedness` | answers backed by tool evidence | numbers absent from every tool result are penalised; abstaining on answerable tasks costs −0.25 so the model doesn't learn to refuse |
| `tool_validity` | real tool names, well-formed JSON arguments | — |

AgentTune's built-in `search_grounding_reward` and `exploration_reward` pay *more* for *more*
calls — the opposite of our goal — so we don't use them. AgentTune folds weighted rewards into
one, so log the terms separately when calibrating.

## Team workflow

- Work is tracked as GitHub Issues on a Project board generated from `docs/WORKPLAN.md`
  (`bash scripts/setup_github_project.sh --apply`).
- Branch per issue (`cg/12-arg-classifier`), PR into `main`, CI must pass, one review.
- Every run gets a config file, an `EXPERIMENTS.md` row and a committed `metrics.json`.
- Models and datasets go to the team HF org, not git.

## Licences

AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) are source-available for research and
non-commercial use, not OSI open source. Check the Tiny Aya and Aya Expanse licences before
publishing weights or teacher outputs. Code in this repo: pick a licence before going public.
