# lean-agent — Lexsi Labs AI Safety & Alignment Hackathon, Track 3

**Thesis:** reinforcement learning on real tool-calling trajectories makes a small Cohere
open-weights agent **use fewer, better tool calls and hallucinate less, without lowering
task success.**

Problem statements addressed (Track 3): *Efficient trajectories*, *Reliable tool calling*,
and *Reward design and reward-hacking defences*.

```
 eval split ──► collect.py (base) ──► evaluate.py ─┐
                                                   ├─► compare.py ──► before/after table
 train split ─► train.py (AgentTune GRPO) ─► adapter ─► collect.py (tuned) ─► evaluate.py ─┘
                    ▲                                              (our metrics + AuditKIT scorers)
            rewards.py: correctness · efficiency · groundedness · tool validity
```

## Stack

| Piece | Used for |
|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | GRPO on multi-turn tool-calling rollouts (`create_agentic_trainer`) |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Independent agent scorers: `redundant_tool_calls`, `agent_loop_detection`, `tool_call_validity`, `tool_selection` |
| `CohereLabs/tiny-aya-global` | Base model (gated — accept the licence on the Hub) |
| Hugging Face Hub | Adapters, eval data, logs |
| W&B or Trackio | Training curves |

## Quick start

```bash
git clone <this repo> && cd lean-agent
python3.12 -m venv .venv && source .venv/bin/activate
cp .env.example .env            # fill HF_TOKEN etc.; then: set -a; source .env; set +a
hf auth login                   # and accept the Tiny Aya licence on its model page

make setup-dev && make test     # laptop: rewards/metrics/tests, no GPU needed
make setup                      # GPU box: also installs AgentTune from source
make smoke                      # 10 training steps, checks the plumbing
make baseline                   # roll out the base model on the held-out eval split
make train                      # GRPO with our reward stack
make after evaluate compare     # tuned rollouts → metrics → before/after table
```

## Repo layout

```
configs/                 one YAML per experiment (copy, don't edit in place)
src/leanagent/
  trajectory.py          parses trajectories (OpenAI tool_calls AND Cohere/AgentTune JSON fallback)
  rewards.py             the reward stack — the core of the project
  data.py                BioGRID QA + unanswerable probes for hallucination
  tools.py               the agent's tools (read-only SQL to start)
  rollout.py             eval-time harness, identical for base and tuned
  metrics.py             headline metrics
scripts/                 train · collect · evaluate · compare · setup_github_project.sh
tests/                   offline tests (no GPU) — run in CI on every PR
docs/WORKPLAN.md         milestones + task breakdown → GitHub issues
docs/EXPERIMENTS.md      one row per run
results/*.metrics.json   committed evidence for the demo
```

## Reward design (and how we defend against hacking)

| Reward | Pays for | Hacking defence |
|---|---|---|
| `correctness` | right final answer, or abstaining when gold is `UNANSWERABLE` | — |
| `efficiency` | few calls, no repeated identical calls, no failed calls, within step budget | **paid only if the answer is correct *and* backed by ≥1 successful tool result** — otherwise "answer yes without calling anything" would be optimal on a yes/no task |
| `groundedness` | answers backed by tool evidence; abstaining on unanswerable questions | numbers in the answer that appear in no tool result are penalised; abstaining on *answerable* questions costs −0.25 so the model doesn't learn to refuse |
| `tool_validity` | real tool names, well-formed JSON arguments | — |

Note: AgentTune's built-in `search_grounding_reward` and `exploration_reward` pay *more* for
*more* calls, the opposite of our goal, so we don't use them. AgentTune folds weighted rewards
into one combined reward, so log the terms separately when calibrating weights (WORKPLAN W2).

## Metrics we report

`task_success`, `calls_per_success`, `avg_tool_calls`, `redundant_call_rate`,
`failed_call_rate`, `malformed_call_rate`, `hallucination_rate`,
`correct_abstention_unanswerable`, plus AuditKIT's scorers as an independent check (so we're
not only grading ourselves with our own reward code). Base and tuned models are rolled out
with the same harness, prompt, eval split and greedy decoding.

## Team workflow

- Work is tracked as GitHub Issues on a Project board, generated from `docs/WORKPLAN.md`
  (`bash scripts/setup_github_project.sh --apply`).
- Branch per issue (`cg/12-dup-penalty`), PR into `main`, CI must pass, one review.
- Every run gets a config file, an `EXPERIMENTS.md` row and a committed `metrics.json`.
- Models and datasets go to the team HF org, not git.

## Licences

AgentTune (LSAL v1.1) and AuditKIT (LSAL v1.2) are source-available for research and
non-commercial use, not OSI open source. Check the Tiny Aya model licence before publishing
weights. Code in this repo: choose a licence before going public (e.g. Apache-2.0).
