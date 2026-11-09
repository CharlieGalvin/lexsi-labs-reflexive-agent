# scripts/

Command-line tools, one folder per job. Run them through `just` (the command names don't change
when a file moves). Shared logic lives in `src/reflexive/`; a script only parses arguments and
prints. **Cost** is the API spend of one run; "none" means no API calls.

## data/: the Curator's data

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `make_splits.py` | `just splits-check` | Froze `configs/splits.json` (train / dev / test); `--check` re-draws it from its seed and compares | none |
| `build_golden.py` | `just golden` | Golden trajectories v1 from the teacher's fix run → `data/golden/v1/`, every record checked | none |
| `build_hard_subset.py` | `just hard-subset` | The hard subset: every train task grouped as healable / teacher-unsolved / solved by Aya, with Arm A's failure labels → `data/golden/hard/` | none |
| `publish_golden.py` | `just golden-publish --tag v1.1` | Checks `data/golden/v1/` (and, for any tag but `v1`, the hard subset), writes the data card, uploads it **private** to Hugging Face in one commit and tags it (`--dry-run` first) | none |

## arms/: the experiment arms

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `rollout.py` | `just rollout A retail` | Runs an arm (`configs/arms.yaml`) on a domain's split → `results/episodes/arms/` | Aya + customer: Arm A on telecom train (74 tasks) was $11.25 |
| `spot_check_hallucination.py` | `just hallucination-spot-check` | Samples the classifier's findings to check by hand; with `--labels`, reports precision | none |

## teacher/: the teacher bake-off and the fix run (SEL-58, SEL-45)

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `bakeoff.py` | `just teacher-sample` | Phase 1: candidate teachers on a fixed 30-task sample → `results/episodes/sample-*.jsonl` | the whole slate was ~$26 |
| `fixes.py` | `just teacher-fixes` | Phase 2: the chosen teacher on every train task, up to 3 attempts → `results/fixes/` (`--report`: no API calls) | GLM-5.3-Flash on all 178 tasks was $11.20 |
| `bakeoff_report.py` | `just bakeoff-report` | Phase 1 statistics (Wilson ranges, McNemar) and projected cost | none |
| `bakeoff_plot.py` | (none) | The quality-vs-cost chart from `bakeoff_report.py --json` (needs `uv pip install -e ".[reports]"`) | none |
| `phase2_report.py` | (none) | Phase 2: GLM-5.3-Flash vs Qwen3.8 27B, paired (`--plot` needs `.[reports]`) | none |

## heal/: the heal loop (SEL-47)

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `cycle.py` | `just heal --fixes-only --sample 3` | Heal cycles end to end: the student on train tasks, the teacher fixing each failing turn, training on the fixes, the gate on all 36 dev tasks, the adapter swapped only on a pass; an audit record per cycle → `results/heal/`. `--fixes-only` stops before training (works with Cohere's Aya); a full cycle needs the student served by vLLM | the student + customer per conversation, as `rollout.py`; the teacher ~$0.001 a failure per attempt |

## student/: the student

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `review.py` | `just student-review` | Where Aya goes wrong: failure types, counts, and an HTML page of every conversation | none |

## tau2/: τ²-bench setup and checks

| Script | `just` command | What it does | Cost |
|---|---|---|---|
| `inspect_tasks.py` | `just tau2-inspect` | Splits, prompt sizes and context fit per domain | none |
| `first_run.sh` | `just tau2-first-run` | One official τ²-bench run, then its cost report | ~$0.70 |
| `cli.py` | (used by `first_run.sh`) | τ²-bench's own CLI, with our pricing for OpenRouter models | (as above) |
| `run_report.py` | (used by `first_run.sh`) | Pass rate and cost of an official τ²-bench run | none |
| `check_scorer.py` | `just tau2-check-scorer`, `just tau2-check-gold` | Our scorer against τ²-bench's own results, and on every task's correct actions | none |
| `episode_check.py` | `just tau2-episode-check` | GPT-4.1 drives our conversation wrapper on 5 telecom tasks | ~$0.70 |
| `dump_tool_schemas.py` | `just tool-schemas` | Saves τ²-bench's tool schemas for the classifier (`--check` verifies) | none |

## archive/

Early probes of Aya's tool-call format (`just teacher-probe`, `just teacher-check`), kept for the
record. See `archive/README.md`.
