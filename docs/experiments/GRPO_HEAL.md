# Heal GRPO (arm H): the teacher's fixes at the student's Tool / Arg / Loop failures (SEL-75)

**Status: plan, for the team to check (11 Nov).** Nothing here has run on the GPU yet. The rule in
§6 is fixed before any H result.

## 1. Why a second GRPO run

The first run (G, SEL-21/SEL-49) failed validation, 0.5/30 against SFT v2's 17.0. It trained on the
**teacher's own clean conversations, and only on their tool-call turns**, so GRPO rewarded calling
tools and the model stopped replying to customers. It also never looked at a failure.

The project's loop is different (README §1): **Detect & Classify → teacher fix → retrain**. The
student fails, Chiara's classifier labels the failure **Tool**, **Arg** or **Loop**, the teacher
rewrites the failing turn, and training learns from that fix. H does exactly that, with one reward
per failure type, optimised for Tool and Arg. Loop counts for less, because the runtime monitor
(SEL-62, SEL-69) is there to stop loops as they happen.

## 2. The logic flow

```
 hard_subset.jsonl (golden v1.1)                          the student's failed conversations
   Arm A on the 112 healable train tasks                   (as it saw them; optional: SFT v2's
            │                                               own failures, --episodes)
            ▼
 ① DETECT & CLASSIFY   reflexive.hallucination.classify_record           (Chiara, SEL-12)
     every finding: TOOL / ARG / LOOP, with its turn
            │  first finding of EACH type = a heal point (same turn → one point, both labels)
            ▼  178 points from 108 conversations: TOOL 47, ARG 103, LOOP 63
 ② SYNTHESISE          reflexive.heal.PrefixTeacher + check_fix           (Chiara, SEL-47)
     GLM-5.3-Flash sees the conversation up to that turn and rewrites that one turn;
     accepted only if it's a real turn, doesn't give up, and has no Tool/Arg/Loop finding
            │  just heal-fixes → data/heal/fixes.jsonl (reagent.sft/v1, labelled)
            ▼
 ③ GRPO PROMPTS        reflexive.heal_data.build                          (this ticket)
     one prompt per fix: the student's conversation up to the failing turn (Aya's format),
     the fix as the reference turn (a tool call OR a reply) + what the prompt already holds
     (earlier calls, which failed, earlier replies) + the student's failed turn
     + ANCHORS: the teacher's ordinary turns (replies and calls), one per fix, label NONE
     10% of the fixes' tasks held out (whole tasks) for the diagnostic
            │  just grpo-heal-data → data/train/grpo-heal.jsonl (+ .heldout, .report)
            │  just grpo-heal-reward-check → every teacher turn at the max; every failed turn
            ▼                                  fails its own type's reward
 ④ RETRAIN             scripts/grpo/train_single_turn.py, configs/train/grpo_heal_aya32b.yaml
     from SFT v2's adapter; for each prompt GRPO samples 8 turns and scores each with
     five reward functions (§4); the adapter moves towards the samples above their group's mean
            │  just grpo-train configs/train/grpo_heal_aya32b.yaml --push
            ▼
 ⑤ VERIFY              held-out diagnostic (just grpo-heldout …), then arm H on the
     30 validation tasks against S2 with the rule in §6; dev; test only if it passes
```

## 3. Data

| | Count (from the hard subset) | Where the student went wrong |
|---|---:|---|
| **TOOL** points | 47 | all `customer_tool`: telecom's phone tools, which the customer must run |
| **ARG** points | 103 | `placeholder` 52 (e.g. `<order_id>`), `ungrounded` 48 (an invented ID or email), schema 3 |
| **LOOP** points | 63 | `repeated_failed_call` 58, `repeated_message` 3, `repeated_call` 2 |
| Points (turns) | 178 | 213 labels: 35 turns carry two types |

Expected after the teacher: roughly 150–170 accepted fixes (the SEL-47 heal run accepted 6/6 on its
first try; the rate on 178 is unknown), plus as many anchor turns. 100 steps × 2 prompts = 200
prompts, about one pass over the training prompts.

## 4. Rewards: one per failure type (`reflexive.heal_rewards`)

Each sampled turn gets five scores, 0–1, passed to TRL as **five separate reward functions**, so
W&B charts each one (`rewards/tool/mean`, …). The total is their weighted sum, at most 4.0.

| Reward | Weight | Scores 1 when | Built on |
|---|---:|---|---|
| `tool` | **1.0** | every call names one of the agent's own tools | the classifier's TOOL check |
| `arg` | **1.0** | no placeholder, invented identifier or schema error in the arguments (share of calls); an unreadable `Action:` scores 0 | the classifier's own `_arg_findings` |
| `loop` | 0.5 | no repeat of a call that already failed, no third identical call or reply, not the failed turn again (unless the teacher's fix does the same) | the classifier's LOOP kinds, read from the prompt |
| `fix` | 1.0 | the teacher's choice: reply where it replied (1.0); where it called, a call (0.4) with its tools (0.3) and arguments (0.3) | the teacher's fix, or the anchor's turn |
| `guard` | 0.5 | the call is the whole reply; no stop, hand-off or few-word reply unless the teacher did | as the first run |

**Checked on the real data** (with a stand-in teacher; rerun on the real fixes with
`just grpo-heal-reward-check`): every teacher turn scores 4.0, and every student failed turn fails
its own type's reward: TOOL 42/42, ARG 92/92, LOOP 54/54. Every loop is caught by the loop rules
themselves, not only by "same as the failed turn".

**Why G can't happen again.** A reply now scores full marks wherever the teacher replied (the
anchors and most Arg fixes, which ask the customer for the missing ID), and loses `fix` wherever
the teacher called a tool. `just grpo-heldout` now reports `kind_match` (replied or called as the
teacher did) and the share of replies, overall and per failure type: the number that would have
caught G before serving it.

## 5. Assumptions: please check these

1. **The failures are Arm A's (base Aya's), not SFT v2's.** v2 rarely makes Tool or Arg errors
   (4 of 60 validation conversations), so its own failures can't teach those types; base Aya's can
   (ARG 103, TOOL 47). The prompts are therefore off-policy for v2: they show base Aya's earlier
   turns. SFT v2's own failures on the train pool (`S2-*-trainpool.jsonl`, SEL-73) can be added with
   `--episodes`; they are mostly Loop and decision errors.
2. **One point per failure type per conversation:** the first Tool, the first Arg and the first
   Loop finding. Later findings of the same type usually follow from the first.
3. **Fixes are accepted by Chiara's `check_fix`** (no Tool/Arg/Loop finding, no giving up), not by
   re-running the task: a fix is "not a hallucination", not "the task now passes" (as in SEL-47).
4. **Every failure reward applies to every prompt,** whatever its label: an answer at a Loop point
   can still invent an argument. The label decides the data mix; the weights decide what's
   optimised. (Alternative: score each prompt only on its own type's reward.)
5. **"Optimise for Tool and Arg"** is implemented as weights (tool 1.0, arg 1.0, loop 0.5) plus the
   mix (150 of the 213 labels are Tool or Arg). We don't require Loop to improve: the runtime
   monitor handles loops.
6. **A reply is scored on kind only** (reply versus call), not its wording: there's no one right
   reply, and an LLM judge would cost per sample.
7. **Anchors, one per fix,** from the teacher's turns on the same train tasks (replies and calls),
   so the model keeps doing ordinary turns right. `--anchors-per-fix` changes the ratio.
8. **The same start and settings as G:** SFT v2's adapter, 100 steps, 8 samples, learning rate
   1e-5, beta 0.04, temperature 1.0. H differs from G only in data and reward, so the comparison
   is clean. G's low KL (under 0.004) suggests these settings move the model little; we keep them
   rather than tune two things at once.
9. **No validation, dev or test task is ever a point:** only the hard subset's `healable` group
   (the tasks SFT trained on), checked again against the splits. 10% of those tasks are held out
   for the diagnostic.
10. **The loop reward reads failures from the prompt text** (`result: Error`), as the classifier
    reads them from the tool messages; it can miss a failure reported without "Error".

## 6. Evaluation and the rule (fixed 11 Nov, before any H result)

Same rule as G (SEL-49), so H is judged on the same terms. On the 30 validation tasks, 2 samples
each (`just arm-val H configs/sft/arms.grpo_heal.yaml`, `just arm-samples S2 H --split val`),
**H is better than SFT v2 if all three hold:**

1. its total (mean of 2 samples) is **19.5 or more** (v2's 17.0 plus more than its sample spread);
2. **no domain drops by more than 1 task** against v2: retail ≥ 5.5, airline ≥ 1.5, telecom ≥ 7.0;
3. **Tool/Arg hallucination is no higher than v2's:** at most 4 of 60 conversations.

Reported alongside, not part of the rule: the Tool, Arg and Loop rates per domain against v2
(`just arm-metrics`), and loops, steps and repeated calls (`just arm-compare`). If H passes, it
runs on test (SEL-26) and whether it replaces v2 as Arm B is agreed with the team. If not, it is
reported on validation and dev.

## 7. Commands

```bash
# laptop or pod, no GPU (OPENROUTER_API_KEY for the teacher)
just heal-fixes --dry-run                       # 178 points: TOOL 47, ARG 103, LOOP 63
just heal-fixes --limit 5                       # smoke: look at data/heal/fixes.jsonl
just heal-fixes                                 # all points (resumes; ~$1 of teacher calls)
just grpo-heal-data                             # → data/train/grpo-heal.jsonl, .heldout, .report
just grpo-heal-reward-check                     # must print every teacher turn at the maximum

# GPU pod
just grpo-heldout configs/train/grpo_heal_aya32b.yaml --data data/train/grpo-heal.jsonl --limit 40 --out runs/heal-precheck
just grpo-train configs/train/grpo_heal_aya32b.yaml --dry-run
just grpo-train configs/train/grpo_heal_aya32b.yaml --max-steps 3 --report-to none
just grpo-train configs/train/grpo_heal_aya32b.yaml --push --force && just grpo-heldout configs/train/grpo_heal_aya32b.yaml
ADAPTER= EXTRA_ADAPTERS="heal=runs/grpo-heal-aya32b-v2" just sft-serve      # then, in another tab:
just arm-val H configs/sft/arms.grpo_heal.yaml && just arm-samples S2 H --split val
```

The precheck (SFT v2 on 40 training prompts) shows two things before spending GPU hours:
`all_same` (groups with nothing to learn) and `kind_match` (how often v2 already replies or calls
as the teacher does), by failure type.

**Cost:** teacher about $1; GPU about 3.5 h of training plus about 1.5 h of checks and evaluation
on an H100 ($3.99/h): about $20.
