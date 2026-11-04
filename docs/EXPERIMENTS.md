# Experiment log

One row per run. Link the W&B run, the config file, the HF adapter and the metrics file.
Never overwrite a row; add a new one. Negative results are results.

| Date | Run | Owner | Config | Change vs. previous | task_success | calls/success | halluc. rate | Notes / links |
|---|---|---|---|---|---:|---:|---:|---|
| 2026-11-0? | base | | | untuned baseline (Arm A) | | | | |
| 2026-11-04 | arm-A-telecom-train | Chiara | `configs/arms.yaml` arm A @ 5da8ae4 | untuned baseline (Arm A): Aya Expanse 32B, telecom train (74 tasks × 1) | 5/74 = 6.8% (95% 2.9–14.9%) | 242 | 98.6% (73/74, Tool or Arg) | Heuristic flags, not the SEL-12 classifier: Tool 73 (customer-side tools like `check_network_status`), Arg 21, Loop 46 conversations. Ends: 43 user_stop, 25 too_many_errors, 6 max_steps; 0 overflows. $11.25. [metrics](../results/arm-A-telecom-train.metrics.json). 65 conversations re-run after Trial-key rate limits |
