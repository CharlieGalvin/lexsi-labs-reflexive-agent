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
first try; the rate on 178 is unknown), plus **half as many anchor turns** (`--anchors-per-fix 0.5`):
about 140–150 fixes and 70–75 anchors to train on once 10% of the tasks are held out. 100 steps
× 2 prompts = 200 prompts, **about 90% of one pass, two-thirds of them fixes** (some 130 fixes,
roughly 30 of them Tool). That is still a modest heal signal; we say so rather than add steps.
*(Changed 11 Nov after Chiara's review, before any result: anchors were 1 per fix, which left
only about 100 fixes in the 200 prompts.)*

## 4. Rewards: one per failure type (`reflexive.heal_rewards`)

Each sampled turn gets five scores, 0–1, passed to TRL as **five separate reward functions**, so
W&B charts each one (`rewards/tool/mean`, …). The total is their weighted sum, at most 4.0.

| Reward | Weight | Scores 1 when | Built on |
|---|---:|---|---|
| `tool` | **1.0** | every call names one of the agent's own tools | the classifier's TOOL check |
| `arg` | **1.0** | no placeholder, invented identifier or schema error in the arguments (share of calls); an unreadable `Action:` scores 0 | the classifier's own `_arg_findings` |
| `loop` | 0.5 | no repeat of a call that already failed, no third identical call or reply, not the failed turn again (unless the teacher's fix does the same) | the classifier's LOOP kinds, read from the prompt |
| `fix` | 1.0 | where the teacher called: a call (0.4) with its tools (0.3) and arguments (0.3); where it replied: a reply (0.2) and what it says (0.8, word overlap with the teacher's reply) | the teacher's fix, or the anchor's turn |
| `guard` | 0.5 | the call is the whole reply; no stop, hand-off or few-word reply unless the teacher did | as the first run |

**The failure rewards count only for the teacher's kind of turn** (Chiara's review, 11 Nov). A
reply makes no call, so it can't fail `tool`, `arg` or `loop`. Without this rule, a generic reply
scored 3.0 of 4.0 wherever the teacher called a tool, more than any call with one invented
argument, and 4.0 wherever it replied, whatever it said: the reward could teach "always ask" as
G's taught "always call". Now:

| Sampled turn | Teacher replied | Teacher called a tool |
|---|---:|---:|
| The teacher's own turn | 4.0 | 4.0 |
| A reply naming what the teacher's names (IDs, items, amounts) | up to 4.0 | 0.5 |
| A generic reply ("Could you please provide more details?") | 3.2 | 0.5 |
| Right tool, right arguments | 0.5 | 4.0 |
| Right tool, one invented ID | 0.5 | about 2.6–2.7 |

What a reply says is scored as the overlap (F1) of its content words with the teacher's reply:
IDs, names and amounts count, filler ("please", "details", "one moment") doesn't. Where the
teacher replied, a group's samples are mostly replies, so this is what separates them. A call that
names only tools the agent doesn't have scores 1.0 on `arg` (it has no arguments to check against
a schema); `tool` scores it 0.

**Checked on the real points** (with a stand-in teacher; rerun on the real fixes with
`just grpo-heal-reward-check`, which must pass):

* every teacher turn scores 4.0;
* every student failed turn fails its own type's reward: TOOL 42/42, ARG 92/92, LOOP 54/54. Every
  loop is caught by the loop rules themselves, not only by "same as the failed turn";
* **canned replies can't game it:** three generic replies score 0.5 where the teacher called and
  3.2 where it replied (must stay at most 1.0, and at least 0.6 below the teacher's 4.0).

**The drift guard** (`drift_guard` in the config). Training logs `heal/reply_excess`: the samples'
share of replies minus the teacher's on the same prompts. The first 10 steps are SFT v2's baseline.
From step 20, if the last 10 steps' average is more than 10 points further from the teacher than
the baseline was, training stops and saves the adapter. That catches G's failure (replying far
less) and the opposite (far more). `just grpo-heldout` also reports `kind_match` and the reply share
per failure type, and can be run on the saved step-25 and step-50 checkpoints afterwards
(`--adapter step25=runs/grpo-heal-aya32b-v2/checkpoints/checkpoint-25`).

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
4. **Every failure reward applies to every prompt,** whatever its label, when the turn is the
   teacher's kind: an answer at a Loop point can still invent an argument. The label decides the
   data mix; the weights decide what's optimised.
5. **"Optimise for Tool and Arg"** is implemented as weights (tool 1.0, arg 1.0, loop 0.5) plus the
   mix (150 of the 213 labels are Tool or Arg). We don't require Loop to improve: the runtime
   monitor handles loops.
6. **A reply is scored on its content words against the teacher's reply,** not by a judge: an LLM
   judge would cost per sample, and word overlap is enough to tell a specific reply from a generic
   one. It can't tell two different but equally good replies apart.
7. **Anchors, half as many as fixes,** from the teacher's turns on the same train tasks (replies and
   calls), so the model keeps doing ordinary turns right. `--anchors-per-fix` changes the ratio.
8. **The same start and settings as G:** SFT v2's adapter, 100 steps, 8 samples, learning rate
   1e-5, beta 0.04, temperature 1.0, so H differs from G only in data and reward and the
   comparison is clean. *That's the only reason.* G's low KL on its training prompts didn't stop
   its behaviour changing a lot elsewhere (17.0 → 0.5 on validation), so low KL is not a safeguard;
   the drift guard and the step-25 and step-50 checkpoints are.
9. **No validation, dev or test task is ever a point:** only the hard subset's `healable` group
   (the tasks SFT trained on), checked again against the splits. 10% of those tasks are held out
   for the diagnostic.
10. **The loop reward reads failures from the prompt text** (`result: Error`), as the classifier
    reads them from the tool messages; it can miss a failure reported without "Error".

## 6. Evaluation and the rule (fixed 11 Nov, before any H result)

Same rule as G (SEL-49), so H is judged on the same terms. On the 30 validation tasks, 2 samples
each, **H is better than SFT v2 if all three hold:**

1. its total (mean of 2 samples) is **19.5 or more**. v2's two samples passed **16 and 18** (mean
   17.0, spread 2), so the bar is above 17.0 + 2 = 19.0, and 19.5 is the next value a mean of two
   samples can take;
2. **no domain drops by more than 1 task** against v2 (6.5 / 2.5 / 8.0): retail ≥ 5.5,
   airline ≥ 1.5, telecom ≥ 7.0;
3. **Tool/Arg hallucination is no higher than v2's:** at most 4 of 60 conversations.

**How it's scored, as v2's 17.0 was:** the runs' own verdicts, per sample (`just arm-samples S2 H
--split val`), not the oracle; Tool/Arg by the classifier, any Tool or Arg finding in a
conversation (`just arm-tool-arg H --split val`, `classify_record`, as for v2's 4 of 60). Crashed
conversations are re-run, as for v2.

**Serving check** (G's evaluation skipped it): before H's validation, SFT v2 is served next to H
and run on the 13 retail validation tasks, 1 sample (`--out …/S2check-retail-val.jsonl`). v2
averaged 6.5 of 13 there; a result far below that means the serving, not H, is off.

**If the drift guard stops training,** H is the adapter saved at that point, and the run is
reported as stopped by the guard; the rule applies to that adapter. Only that one adapter is
judged: no checkpoint is chosen by its results.

**Dev, whatever validation says** (SEL-49, the M2 exit criterion: does GRPO beat SFT v2 and Arm A
on dev?): H on the 36 dev tasks, 2 samples each, oracle-scored; `just arm-compare` H against S2
and H against Arm A, per domain, with 95% ranges; `just gate-check S2 H`.

Reported alongside, not part of the rule: the Tool, Arg and Loop counts per domain against v2
(`just arm-tool-arg S2 H`), and loops, steps and repeated calls (`just arm-compare`). If H passes,
it runs on test (SEL-26) and whether it replaces v2 as Arm B is agreed with the team. If not, it is
reported on validation and dev.

**If H isn't finished for the 14 Nov report:** the report carries SFT v2 as Arm B (its test run,
SEL-26), G as a negative result with its cause, and H as in progress with this plan and whatever
is done (fixes, the reward checks, training curves).

## 7. Commands

```bash
# laptop or pod, no GPU (OPENROUTER_API_KEY for the teacher)
just heal-fixes --dry-run                       # 178 points: TOOL 47, ARG 103, LOOP 63
just heal-fixes --limit 5                       # smoke: look at data/heal/fixes.jsonl
just heal-fixes                                 # all points (resumes; ~$1 of teacher calls)
just grpo-heal-data                             # → data/train/grpo-heal.jsonl, .heldout, .report
just grpo-heal-reward-check                     # teacher turns at the maximum; canned replies can't game it

# GPU pod
just grpo-heldout configs/train/grpo_heal_aya32b.yaml --data data/train/grpo-heal.jsonl --limit 40 --out runs/heal-precheck
just grpo-train configs/train/grpo_heal_aya32b.yaml --dry-run
just grpo-train configs/train/grpo_heal_aya32b.yaml --max-steps 3 --report-to none
just grpo-train configs/train/grpo_heal_aya32b.yaml --push --force && just grpo-heldout configs/train/grpo_heal_aya32b.yaml
python -c "from huggingface_hub import snapshot_download as s; s('reagent-lexsi/sft-aya32b-r16-v2', local_dir='runs/sft-aya32b-r16-v2')"   # no huggingface-cli on the pod
ADAPTER= EXTRA_ADAPTERS="heal=runs/grpo-heal-aya32b-v2 sft2=runs/sft-aya32b-r16-v2" just sft-serve   # then, in another tab:
just arm-val S2 configs/sft/arms.sft2.yaml --domains retail --samples 1 --out results/episodes/arms/S2check-retail-val.jsonl
just arm-val H configs/sft/arms.grpo_heal.yaml && just arm-samples S2 H --split val && just arm-tool-arg H --split val
```

The precheck (SFT v2 on 40 training prompts) shows two things before spending GPU hours:
`all_same` (groups with nothing to learn) and `kind_match` (how often v2 already replies or calls
as the teacher does), by failure type.

**Cost:** teacher about $1; GPU about 3.5 h of training plus about 1.5 h of checks and evaluation
on an H100 ($3.99/h): about $20.
