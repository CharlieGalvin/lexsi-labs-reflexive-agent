# The runtime monitor's `no_progress` check on retail and airline (SEL-69)

**Status:** final for the saved Arm A rollouts (9 Nov). The guard ablation (the rest of
SEL-69) will be added here.

**Question.** The runtime monitor (SEL-62, `reflexive.monitor`) raises `no_progress` after 3
agent turns in a row with no tool call, each repeating something the agent already said. On
telecom it fired on all 7 conversations that looped to τ²-bench's step limit and on none of
the 7 that passed. Does it behave the same on retail and airline: fire on the loops and never
on a pass?

**Answer.** It never fires on retail or airline, because there is nothing to catch. None of the
104 Arm A conversations reached the step limit. No turn repeated earlier text more than once in
a row, and the rule needs three. So there are **no false alarms** (0 of 14 passes, 0 of 90
failures), but this data **can't test recall** on these domains. Telecom, replayed the same way,
reproduces the SEL-62 result exactly.

## Method

No new conversations. `scripts/arms/monitor_replay.py` feeds each saved record's `agent_view`
(the conversation as the agent saw it) to `Watcher("act", tools)` one agent turn at a time, as a
live run would: after the agent's turn and its tool results. The tools come from
`domain_tools(domain)`. For each conversation it records where `no_progress` fires, what the
guard would have done in `act` mode (nudge, then stop), the termination and the pass, and the
longest run of repeated text-only turns.

```
python scripts/arms/monitor_replay.py results/episodes/arms/A-{telecom,retail,airline}-*.jsonl \
  results/episodes/arms/C-telecom-train.jsonl --metrics results/monitor-no-progress-replay.metrics.json
```

The inputs are the Arm A rollouts from SEL-55 (Aya Expanse 32B via Cohere, sample 0), plus the
4 Nov Arm A and C telecom train rollouts as a check of the method. They are git-ignored, like all
episode files.

## Results

| File | Conversations | Step limit | Passed | `no_progress` fired | Longest repeat run | Max agent turns |
|---|---:|---:|---:|---:|---|---:|
| A-telecom-train (check) | 74 | 6 | 5 | 6, all at the step limit | 0 (68), 8 to 67 (the 6 loops) | 100 |
| C-telecom-train (check) | 74 | 1 | 2 | 1, at the step limit | 0 (72), 1, 3 | 99 |
| A-retail-train | 59 | 0 | 8 | 0 | 0 (53), 1 (6) | 27 |
| A-retail-dev | 15 | 0 | 1 | 0 | 0 (15) | 15 |
| A-airline-train | 24 | 0 | 4 | 0 | 0 (21), 1 (3) | 19 |
| A-airline-dev | 6 | 0 | 1 | 0 | 0 (5), 1 (1) | 13 |

- **Telecom reproduces SEL-62.** The 7 loops (A37, A42, A56, A68, A69, A73, C42) fire at the
  same agent turns as `tests/fixtures/no_progress_cases.json`, and nothing else fires.
- **Retail and airline conversations end differently.** They are short (at most 27 agent turns,
  telecom's loops run to about 100), and the customer ends them: retail 66 `user_stop` and
  8 `too_many_errors`, airline 30 `user_stop`. The goodbye loop that fills telecom's step limit
  doesn't happen there.
- **No near misses.** The longest run of repeated text-only turns is 1 in every retail and
  airline conversation. The threshold is 3.
- **No fixture change.** The issue adds false alarms to `tests/fixtures/no_progress_cases.json`,
  and there are none.

## What this means

- **The guard ablation is a telecom question.** In `act` mode the guard only reacts to
  `no_progress`, which never fired on retail or airline. Run there, the guard most likely changes
  nothing, so the ablation runs on telecom dev. Retail and airline are left out for this reason.
  The caveat: a different sample could still loop, but nothing in 104 conversations came close.
- **Recall outside telecom is untested.** A retail or airline loop, if one appears in later runs
  (for example under the healed adapter), should be checked against the rule then.
- The known gap from SEL-62 still holds: a loop that rewords itself each time isn't caught.

Metrics: [`results/monitor-no-progress-replay.metrics.json`](../../results/monitor-no-progress-replay.metrics.json).
