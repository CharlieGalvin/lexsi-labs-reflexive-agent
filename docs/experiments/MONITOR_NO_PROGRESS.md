# The runtime monitor on retail and airline, and the guard ablation (SEL-69)

**Status:** final (9 Nov). Two questions: does `no_progress` behave on retail and airline as on
telecom (first half), and does the guard change anything on its own (the ablation, second
half)? Short answer to both: nothing looped, so the monitor stayed quiet and the guard never acted.

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

## The guard ablation (telecom dev)

**Question.** In `act` mode the guard nudges a `no_progress` loop, then stops it. Does that change
success, loops or steps per task on its own, so its effect can be told apart from Arm B's?

**Answer.** It never acted. In 30 guarded conversations `no_progress` never fired, so there was no
nudge and no stop, and the guarded run is the plain one again. The differences in the table below
are run-to-run noise: none is clear, and this is effectively an A/A test. **The guard's effect is
not measured**, because telecom dev didn't loop this time.

**Method.** Arm A (Aya Expanse 32B via Cohere, GPT-4.1 customer) on the 15 telecom dev tasks, 2
samples each, run twice at the same commit (`e3194b3`), side by side: once plain (`--monitor log`)
and once with the guard (`--monitor act`). Retail and airline were left out because the guard can't act
where `no_progress` never fires (first half). Telecom has no LLM judges, so the scores are
complete. One guarded conversation crashed inside τ²-bench (`unexpected_error`, guard not
involved) and was re-run, so both arms have 30.

```
just rollout A telecom --split dev --samples 2                  # → A-telecom-dev.jsonl
just rollout A telecom --split dev --samples 2 --monitor act    # → A-telecom-dev-guard.jsonl
just arm-compare results/episodes/arms/A-telecom-dev.jsonl results/episodes/arms/A-telecom-dev-guard.jsonl \
  --names arm-A-telecom-dev arm-A-telecom-dev-guard
```

| Metric | A | 95% CI | A + guard | 95% CI | Guard − A (paired) | 95% CI |
|---|---:|---|---:|---|---:|---|
| Success rate | 10.0% | 0.0 to 23.3% | 3.3% | 0.0 to 10.0% | −6.7 pts | −23.3 to +6.7 |
| Hallucination rate (Tool or Arg) | 100% | 100 to 100% | 100% | 100 to 100% | 0 | 0 to 0 |
| Loop rate (classifier) | 70.0% | 53.3 to 86.7% | 80.0% | 66.7 to 93.3% | +10.0 pts | −3.3 to +23.3 |
| Steps per task | 11.3 | 9.7 to 12.8 | 10.0 | 8.4 to 11.8 | −1.2 | −3.0 to +0.6 |
| Redundant-call rate | 27.5% | 20.7 to 33.2% | 34.8% | 28.8 to 40.2% | +7.3 pts | −0.1 to +14.1 |

30 vs 30 conversations, paired on 15 tasks; 95% percentile bootstrap, 10,000 resamples, seed 0.

- **The guard never acted:** 0 `no_progress` fires, 0 nudges, 0 `monitor_stop` endings. Replaying
  both files through `monitor_replay.py` agrees (30 quiet each).
- **No conversation reached the step limit:** at most 25 agent turns (telecom's loops ran to about
  100). Endings: plain 21 `user_stop` and 9 `too_many_errors`; guarded 20 and 10.
- **So the 3 vs 1 passes are noise.** The guard didn't touch either run, and the passes fall on
  different tasks in the two runs (plain: both samples of one airplane-mode/APN/SIM-PIN task and
  one roaming task; guarded: one contract-suspension task). That is the student's run-to-run spread at a pass
  rate of about 1 in 10.
- **The "Loop rate" row is the classifier's Loop** (repeated or failing tool calls), not
  `no_progress`. It's high in both runs and isn't what the guard reacts to.
- **Why no loops this time:** on 4 Nov, 7 of 148 telecom train conversations looped to the step
  limit, and one of those tasks (A69) is a dev task. In today's 60 dev conversations, including
  A69's 4, none did. Loops are rare enough (about 5% of telecom conversations) that 30 per arm
  may contain none.

**What it would take to measure the guard:** run it where loops happen. That means the telecom
train tasks that looped (A37, A42, A56, A68, A69, A73, C42), several samples each, or a larger
telecom sample. The replay above shows what `act` would have done on the 4 Nov loops: nudge all
7, stop 6 (C42 fires only on its last turn).

### What a guarded run records (from Charlie's review of #52)

- **A guard stop has its own ending**, `monitor_stop`. τ²-bench grades it as the agent's stop, but
  `arm_metrics` and `arm-compare` report those runs separately (`monitor_stopped`: conversations,
  passes, and the success rate if those passes count as failures). A `monitor_stop` is never a
  golden trajectory. None happened here.
- **The nudge is kept** in `rec["monitor"]["nudge"]`: its text, the message in the agent's view
  it followed (`after_view_index`, `after_role`) and the agent turn it was shown before. The
  nudge goes into one prompt only, not into τ²-bench's conversation, so this is the only record of
  it. **Reader's note:** the nudge follows the customer's message, so the agent sees two user
  turns in a row (`after_role: "user"`).

Metrics: [plain](../../results/arm-A-telecom-dev-sel69.metrics.json),
[guarded](../../results/arm-A-telecom-dev-guard.metrics.json),
[comparison](../../results/arm-A-telecom-dev-vs-arm-A-telecom-dev-guard.compare.metrics.json).
Cost: $2.96 + $2.51 (customer and student; the crashed attempt's spend isn't counted).
