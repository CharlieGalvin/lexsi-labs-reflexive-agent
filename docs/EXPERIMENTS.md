# Experiment log

One row per run. Link the W&B run, the config file, the HF adapter and the metrics file.
Never overwrite a row; add a new one. Negative results are results.

## Training and evaluation runs (Trainer, Architect)

| Date | Run | Owner | Config | Change vs. previous | task_success | calls/success | halluc. rate | Notes / links |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-11-0? | base | | | untuned baseline (Arm A) | | | | |
| 2026-11-04 | arm-A-telecom-train | Chiara | `configs/arms.yaml` arm A @ 5da8ae4 | untuned baseline (Arm A): Aya Expanse 32B, telecom train (74 tasks × 1) | 5/74 = 6.8% (95% 2.9–14.9%) | 242 | 98.6% (73/74, Tool or Arg) | Heuristic flags, not the SEL-12 classifier: Tool 73 (customer-side tools like `check_network_status`), Arg 21, Loop 46 conversations. Ends: 43 user_stop, 25 too_many_errors, 6 max_steps; 0 overflows. $11.25. [metrics](../results/arm-A-telecom-train.metrics.json). 65 conversations re-run after Trial-key rate limits |
| 2026-11-04 | arm-C-telecom-train | Chiara | `configs/arms.yaml` arm C (`careful/v1`) @ 0253859 | Arm A + fixed "be careful not to hallucinate" instruction; same 74 tasks × 1 | 2/74 = 2.7% (95% 0.7–9.3%) | 417 | 100% (74/74, Tool or Arg) | Paired with A: A-only 5, C-only 2, both 0 (exact McNemar p = 0.45): no detectable gain. Tool 74, Arg 18, Loop 54 conversations; customer-side tool calls unchanged (412 vs 422). Gives up sooner: 24 transfers to a human (A: 9), 834 student calls (A: 1,209). $6.73. [metrics](../results/arm-C-telecom-train.metrics.json) |

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
