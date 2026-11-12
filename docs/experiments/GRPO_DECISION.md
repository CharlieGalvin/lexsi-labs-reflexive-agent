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

## H2's training (12 Nov, 01:12–03:38)

Stopped by the drift guard at **step 53 of 100**: the reply share fell 0.17 below the teacher's over
the last 10 steps (−0.06 at the start; limit 0.10 more). Peak 56 GB, 2.75 min a step. Over the
training samples, `decision` didn't rise (0.97 in steps 0–7, between 0.81 and 0.97 after, 0.83 in
steps 48–52), wrong or unexpected changes came in spikes, and **calls went from about 50–57% of turns
to 80%** in the last steps. The step-53 adapter is the one judged on validation, as fixed above.

**Why, as far as we can tell:** the decision reward paid 1.0 for making the teacher's change and
0.4 for holding off (asking first, replying), so it pushed towards acting; and v2 already scored
0.97 on the teacher's decision points, so there was little else to learn from.

## H3: wrong-only reward, at the student's own wrong changes

**Fixed 12 Nov, 08:40, before any H3 result and before H2's validation result.**

Two changes from H2 (`configs/train/grpo_decision_h3_aya32b.yaml`):

1. **Wrong-only decision reward** (`mode: wrong_only` in the data): a state change matching no
   pending expected change scores 0; everything else (making a pending change, asking first,
   replying) scores 1. Nothing pays for acting, so the only lesson is not to change the wrong thing,
   which is most of v2's failures (about 29 of 55: wrong arguments or a change nobody asked for).
2. **Prompts at v2's own wrong changes:** SEL-73's teacher fixes on SFT v2's train-pool
   conversations (`onpolicy-fixes.jsonl`, the turns where v2 made a change it shouldn't have), each
   twice, plus H2's prompts sampled to **100 in all**; **50 steps** (a checkpoint every 10). Built
   with `just grpo-decision-data --name grpo-decision-h3 --wrong-only --onpolicy <file> --total 100`.

Everything else as H2. **Rule:** the same three parts as H2 above (validation total ≥ 19.5; no
domain more than 1 below v2; Tool/Arg at most 4 of 60), on the adapter at the end of training or
where the drift guard stops it. Same 22:00 cut-off. H3 is the eighth model scored on these 30
validation tasks: a pass is a hint, and test (as its own arm) the check.

### H3 amended before it ran (12 Nov, 08:50), still before any H3 or H2 validation result

SEL-73's on-policy fixes aren't on the pod's volume, so change 2 above can't be built. H2's own
training logs (`scripts/grpo/diagnose_run.py runs/grpo-decision-aya32b-v2`) gave a better target:

| Where H2's learning signal went | |
|---|---|
| Prompt groups with no variance in the total reward | 21% |
| ... in the decision reward | **76%** (48% on decision prompts) |
| Share of within-group variance, weighted by the reward weights | tool and arg about 56%, decision about 32% |
| Mean tool / arg reward of the samples (temperature 1.0) | 0.83 / 0.82 |
| Reply share, steps 1–10 → 51–53 | 0.54 → 0.08 |

Most of the signal went to invalid calls sampled at temperature 1.0, which SFT v2 rarely makes
when served at temperature 0, and most decision prompts taught nothing. So H3 is instead:
**prompts where H2's samples disagreed about what to change** (decision varied or fell below 1;
from H2's training samples, never validation), each twice, filled to 100 with H2's others
(`scripts/data/build_grpo_focus_data.py`); the **wrong-only** decision reward; decision **3.0**,
tool and arg **0.5**; sampling temperature **0.8**; 50 steps. Rule and cut-off unchanged.
