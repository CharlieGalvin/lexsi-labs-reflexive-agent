# Decision GRPO (arm H2): H plus a reward that checks each state change against the expected actions

**Status: plan and rule fixed 12 Nov, 01:15, before any H2 result.**

## Why

SFT v2's failures on dev and validation (55 conversations; never test) are mostly **wrong decisions
about what to change**, which the hallucination classifier, and so H's Tool / Arg / Loop rewards,
can't see:

| v2's failure | Count |
|---|---:|
| A state change nobody asked for (e.g. `cancel_reservation`, `send_payment_request`) | ~15 |
| The right tool with the wrong arguments (mostly retail item changes) | ~14 |
| An expected change never made (`cancel_reservation`, `modify_pending_order_items`, `enable_roaming`) | 11 |
| Stuck at the step limit / too many tool errors | 10 |
| Actions right, another check failed | 6 |

The classifier flags a Tool, Arg or Loop finding in 18 of the 55 (17 of them loops), and in 8 of
the 77 conversations v2 passed. H fixed loops and left success unchanged; this is why.

## What changes from H (`configs/train/grpo_decision_aya32b.yaml`)

* **A sixth reward, `decision`** (`reflexive.decision_data.decision_score`): every state-changing
  call in the sampled turn must match an expected change still pending at that turn (τ²-bench's
  expected actions and its matching rule). A change nobody asked for or wrong arguments: 0. Where the
  teacher made a change: 1 if the sample makes it, 0.4 if it holds off. Otherwise 1, or 0.5 for a
  pending change made early. Verified against each task's ground truth, not the teacher's wording.
* **Decision prompts:** the teacher's turns where it makes a state change, and the turn before,
  from its verified train conversations, one per heal fix (`just grpo-decision-data`). H's fixes and
  anchors are kept, with the same held-out tasks.
* **Weights:** decision 2.0, fix 0.5 (H: 1.0); tool 1.0, arg 1.0, loop 0.5, guard 0.5 as H.
* Everything else as H: from SFT v2's adapter, 100 steps, the drift guard. Validation tasks are
  excluded from the data (read from `P-<domain>-val.jsonl`; the builder refuses without them).

## The rule (as G's and H's, fixed before any H2 result)

On the 30 validation tasks, 2 samples each, H2 is better than SFT v2 if all three hold:

1. its total (mean of 2 samples) is **19.5 or more** (v2: 17.0, samples 16 and 18);
2. **no domain drops by more than 1 task** against v2 (6.5 / 2.5 / 8.0);
3. **Tool/Arg hallucination no higher than v2's: at most 4 of 60** conversations.

Scored as v2 and H were: `just arm-val H2 configs/sft/arms.grpo_decision.yaml`, then
`just arm-samples S2 H2 --split val` and `just arm-tool-arg H2 --split val`. Only the adapter at
the end of training (or where the drift guard stopped it) is judged. Dev runs whatever validation
says (`just gate-check S2 H2`). Test only if H2 passes, as its own arm beside B, reported whatever it
scores. **Cut-off: if validation isn't scored by 22:00 on 12 Nov, H2 is reported as unfinished.**

## Caveat

This is the seventh model scored on the same 30 validation tasks (v1–v4, SL, G, H), and the rule
was set after looking at v2's dev and validation failures. A pass is a strong hint, not proof; the
test run would be the check.
