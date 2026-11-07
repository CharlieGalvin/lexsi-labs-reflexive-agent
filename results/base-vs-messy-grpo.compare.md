# base vs messy-grpo

Starter task (Messy-ID Lookup), eval split, greedy. Made by `just compare base messy-grpo` from [base](base.metrics.json) and [messy-grpo](messy-grpo.metrics.json).

| Metric | base | messy-grpo | Change |
|---|---:|---:|---:|
| task_success | 53.3% | 55.8% | +2.5 pts |
| tool_ok | 60.8% | 60.8% | +0.0 pts |
| hallucination rate (Arg) | 39.2% | 39.2% | +0.0 pts |
| format | 100.0% | 100.0% | +0.0 pts |
| calls / success | 1.88 | 1.84 | -0.04 |
| correct: `clean` | 19/22 | 19/22 | +0.0 pts |
| correct: `lower_dash` | 0/21 | 2/21 | +9.5 pts |
| correct: `spaced` | 18/21 | 19/21 | +4.8 pts |
| correct: `number_only` | 0/22 | 0/22 | +0.0 pts |
| correct: `spelled` | 10/15 | 9/15 | -6.7 pts |
| correct: `distractor` | 17/19 | 18/19 | +5.3 pts |
