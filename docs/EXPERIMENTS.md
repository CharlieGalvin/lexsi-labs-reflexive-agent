# Experiment log

One row per run. Link the W&B run, the config file, the HF adapter and the metrics file.
Never overwrite a row; add a new one. Negative results are results.

## Training and evaluation runs (Trainer, Architect)

| Date | Run | Owner | Config | Change vs. previous | task_success | calls/success | halluc. rate | Notes / links |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-11-0? | base | | | untuned baseline (Arm A) | | | | |

## Data runs (Curator)

All on τ²-bench v2.0.0 **train** tasks; never test. Pass = τ²-bench's own evaluator. Cost
includes the simulated customer.

| Date | Run | Models | Tasks | Result | Cost | Write-up / data |
|---|---|---|---|---|---:|---|
| 2026-11-03 | Teacher bake-off, phase 1 (database checks only) | 12 via OpenRouter + Aya Expanse 32B + Tiny Aya | 30 (20 retail, 10 airline) | Top tier tied: Qwen3.8 27B 30/30, Sonnet 5.5 29/30, GLM-5.3-Flash 27/30; Aya Expanse 32B 1/30 | ~$26 (all models, incl. reruns) | [phase 1](experiments/TEACHER_BAKEOFF_RESULTS.md); `results/episodes/sample-*.jsonl` |
| 2026-11-03 | Teacher fix run (phase 2), judges on | GLM-5.3-Flash | all 178 | 156/177 first try (88%); 158 tasks solved; 193 clean passes | $11.20 | [phase 2](experiments/TEACHER_PHASE2_RESULTS.md); `results/fixes/glm53f.jsonl` |
| 2026-11-03 | Phase 2 comparison, judges on | Qwen3.8 27B, thinking off | 139 | 94/138 paired (68%); every telecom failure a hand-off | $8.11 | [phase 2](experiments/TEACHER_PHASE2_RESULTS.md); `results/fixes/qwen38nt.jsonl` |
| 2026-11-04 | Student review (Cohere `Action:` format) | Aya Expanse 32B | 30 (10 per domain) | 7/30; calls customer's phone tools, batches dependent calls, invents IDs; 0 unreadable calls | $1.54 | [student review](experiments/STUDENT_REVIEW_AYA.md); `results/fixes/aya.jsonl` |
| 2026-11-04 | Golden trajectories v1 build | (no model calls) | 158 solved | 127 train + 31 dev reference records; all read back exactly | $0 | [golden v1](experiments/GOLDEN_TRAJECTORIES_V1.md); `data/golden/v1/` |

Raw `results/` and `data/` files are gitignored; the datasets go to the team HF org.
