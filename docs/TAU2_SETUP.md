# τ²-bench setup and decisions (SEL-31)

**Decision: pin τ²-bench v2.0.0 and keep all three domains (retail, airline, telecom) for now.**
Telecom is **at risk**: its prompt alone fills ~91% of Tiny Aya's 8,192-token context window.
Rather than drop it on a measurement alone, we let the first baseline runs decide, using the
drop rule below. Everything here is reproducible with `python scripts/tau2_inspect.py` (no API calls).

**Telecom drop rule (checked after SEL-20 / SEL-55, by Thu 5 – Sat 7):** drop telecom if, in
Arm A rollouts on its train tasks, **more than 25% hit the context limit**, or its pass rate is
near zero with context overflow as the main cause. The overflow rate becomes the evidence in
the write-up. If dropped, we fall back to retail + airline (104 train / 60 test).

## Install

```bash
pip install -e ".[bench]"        # τ²-bench v2.0.0 + gym extra (train/test splits)
```

- Pinned to the **v2.0.0** release tag (commit `989d328`), so every teammate and every run uses
  identical tasks, policies and databases.
- Requires **Python 3.12 or 3.13**. On 3.13, τ²-bench crashes on import (`No module named
  'audioop'`) because Python removed that module; the `bench` extra installs the
  `audioop-lts` backport to fix it.
- `pip install git+https://…` downloads the whole repository and can be very slow. If it
  stalls, clone with `--depth 1`, `git checkout v2.0.0`, then `pip install -e ".[gym]"`.

## Splits (from τ²-bench's own `split_tasks.json`)

| Domain | Train | Test | Notes |
| --- | ---: | ---: | --- |
| retail | 74 | 40 | 16 agent tools |
| airline | 30 | 20 | 14 agent tools |
| telecom | 74 | 40 | At risk (context); also a 2,285-task generated pool for extra training |
| **Total** | **178** | **100** | all three; 104 / 60 if telecom is dropped |

Healing (hard subset, teacher fixes, GRPO) and the DSPy prompt (Arm D) use **train only**. All
reported numbers come from the test tasks (100, or 60 without telecom), run with 3–4 trials
each (τ²-bench's pass^k) so the confidence intervals stay usable on a small test set.

## Context fit for Tiny Aya (8,192 tokens)

| Domain | Policy | Tool schemas | Fixed prompt | Gold-path tool I/O (median / p90) | Tasks over 8K on the gold path |
| --- | ---: | ---: | ---: | ---: | ---: |
| airline | ~1,900 | ~2,600 | ~4,500 | ~320 / ~1,300 | 1 of 50 |
| retail | ~1,700 | ~2,900 | ~4,600 | ~970 / ~2,700 | 7 of 114 |
| telecom | ~5,800 | ~1,600 | ~7,400 | ~120 / ~270 | 56 of 114 |

Approximate tokens (characters / 4). "Gold path" = replaying each task's reference actions,
so real rollouts (extra lookups, retries, chat) will be longer: treat these as lower bounds.

**What this means for the harness (SEL-32):**
- Serialise tool results as **compact JSON** (`separators=(",", ":")`). τ²-bench returns
  indented JSON; compacting alone cut retail overflows from 10 to 7 tasks.
- Add a context guard: when the conversation nears 8K, truncate the oldest tool outputs
  first, and log a `context_overflow` flag so it is reported, not hidden.
- Count context overflow as its own failure type in the error analysis; it is not a
  hallucination and should not be "healed".

## Simulated customer (user simulator)

- **Choice:** `gpt-4.1-2025-04-14`, τ²-bench's default user simulator. Using the default keeps
  our Arm A numbers comparable with published τ²-bench results.
- Optional for cheap development runs only: `gpt-4.1-mini`. Never mix simulators between arms.
- **Cost per task:** τ²-bench records `agent_cost` and `user_cost` for every run.
  `bash scripts/tau2_first_run.sh` runs 2 retail train tasks and prints both. Post the numbers
  in SEL-31 / SEL-37 and multiply: (178 train + 100 test × 3 trials × 4 arms) × user cost per task.

## Version note

v2.0.0's English retail and telecom tasks differ from v1.0.1, so v2.0 scores are **not**
comparable with the 1.x leaderboard. Our headline comparison is between our own four arms on
the same test split, so this doesn't affect the experiment. If we want a leaderboard reference
point for Arm A, use the v2.0 text leaderboard.
