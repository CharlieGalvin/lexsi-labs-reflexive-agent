# Heal GRPO (arm H): the teacher's fixes at the student's Tool / Arg / Loop failures (SEL-75)

**Status: done (11 Nov). H fails its validation rule and doesn't beat SFT v2 on dev success; it
halves loops and cuts repeated calls (§8).** The plan below (§1–§7) was fixed before any result.

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

## 8. Results (11 Nov)

**Summary.** H is far above Arm A and close to SFT v2, but not above it: it fails its validation
rule (at most 17.0 of 30 against a bar of 19.5) and loses to v2 on dev success (18.5 against 21.5 of
36; the gate rejects it). What it changes is how it fails: half as many loops as v2 on dev, and in
telecom a sixth of v2's repeated calls (the one significant difference). Tool and Arg hallucination,
already rare in v2, stay where they were.

**Data and training.** The teacher fixed 177 of 178 points (TOOL 47/47, ARG 103/103, LOOP 62/63).
The reward check passed on the real fixes (235 teacher turns at the maximum; every failed turn
failed its own type's reward; canned replies 0.5 where the teacher called, 3.23 where it replied).
SFT v2's precheck had something to learn on almost every prompt (`all_same` 12%, against G's 68%).
Training ran all 100 steps (`b82e182`, 2.21 min/step, peak 64.2 GB); the drift guard didn't fire.
In the training samples, Tool failures fell from 2.5% to 0 after step 40, Arg failures roughly
halved (11% → 6%), and `fix` rose from 0.60 to 0.72; Loop failures in the samples rose (2% → 8–10%),
mostly at Loop points. KL to SFT v2 peaked at about 0.04. Adapter `reagent-lexsi/grpo-heal-aya32b-v2`.

**Held-out check** (30 prompts from 11 held-out train tasks, 4 samples): SFT v2 → H +0.11 reward (95%
range −0.06 to +0.30): TOOL 3.72 → 3.89, LOOP 2.33 → 2.53, ARG 2.84 → 2.94; anchor turns unchanged
(kind match 78% → 78%). No sign of G's collapse.

**Validation** (30 tasks, 2 samples, the runs' own verdicts, as v2's 17.0). Serving check: v2 passed
7 of 13 retail validation tasks on the same server (its earlier mean 6.5), so serving is fine.

| | Retail (13) | Airline (5) | Telecom (12) | Total |
|---|---:|---:|---:|---:|
| SFT v2 | 6.5 | 2.5 | 8.0 | **17.0** |
| H | 5.5 (11/26) | 2.5 (5/10) | ≥ 7.0 (14/20 scored, 4 crashed) | **≥ 15.0, ≤ 17.0** |
| Rule | ≥ 5.5 ✓ | ≥ 1.5 ✓ | ≥ 7.0 ✓ | ≥ 19.5 **✗** |

**H fails part 1**, whatever its 4 crashed telecom conversations would have scored. Part 3 (Tool/Arg
on validation) wasn't printed by the evening run: `just arm-tool-arg H --split val`, no GPU needed.
H doesn't run on test; SFT v2 stays Arm B.

**Dev** (36 tasks, 2 samples, oracle-scored with judges on; SEL-49):

| | Arm A | SFT v2 | H |
|---|---:|---:|---:|
| Success, mean of 2 samples | 2/36 (6%) | **21.5/36 (60%)** (24, 19) | 18.5/36 (51%) (19, 18) |
| Tool or Arg (conversations) | — | 5/72 | 5/72 |
| Loop (conversations) | — | 12/72 | **6/72** |
| Classifier findings per conversation | — | 1.17 | **0.25** |

| H − SFT v2, paired by task (95% range) | Retail | Airline | Telecom |
|---|---:|---:|---:|
| Success | −3.3 pts (−23.3 to +20.0) | −16.7 (−58.3 to +16.7) | −10.0 (−23.3 to +3.3) |
| Tool or Arg | +3.3 (0.0 to +10.0) | 0.0 | −3.3 (−10.0 to 0.0) |
| Loops | −10.0 (−23.3 to +3.3) | −8.3 (−25.0 to 0.0) | −6.7 (−20.0 to +6.7) |
| Steps per task | −0.6 | +0.4 | −3.9 (−8.9 to +0.8) |
| Repeated calls | −7.2 | −1.9 | **−38.5 (−58.8 to −1.0)** |

`just gate-check S2 H` (sample 0): **REJECT**, 66.7% → 52.8%; 2 tasks only H passed, 7 only v2
(McNemar p = 0.18); retail −1, airline −1, telecom −3. Against Arm A the comparison is unpaired
(its dev conversations aren't on the volume; the committed metrics give 2/36): H's 18.5/36 is far
above it.

**What H lost on, against v2** (dev; per-check verdicts and `scripts/arms/failure_diff.py`, 12 Nov):

* **Which checks failed.** Retail: v2 had 3 conversations ended by tool errors or the step limit,
  H none (1 context overflow); wrong final database 11 → 13. Airline: wrong database 5 → 7.
  Telecom: customer's device not fixed (`ENV_ASSERTION`) 4 → 7; step limit and missing actions
  unchanged (3 and 3). Of the 12 conversations v2 passed and H failed (same task and sample), 7
  have a wrong database, 3 an unfixed device, 1 the step limit, 1 a judged message; H's are mostly
  *longer* than v2's (retail/76: 50 messages against 30; telecom: 152 against 62), so it didn't give up.
* **What went wrong in the database** (retail and airline failures, expected actions against what
  the agent did). Missing changes 11 (v2) against 10 (H); wrong details 10 against 10; unrequested
  changes 26 against 26, but of a different kind. **v2's extras are error spirals**: 16 of the 26
  are one failing `modify_pending_order_items` retried 9 and 7 times (retail/112, retail/35) until
  `too_many_errors`. H has none. **H's extras are valid, unrequested state changes**: 12 airline
  `cancel_reservation` calls (task 39: the 3 asked for plus 4 more; 41 and 43: the wrong ones), an
  unrequested `send_certificate` (36), cancelling a retail order that should have been modified
  (78), returning extra items (82).

So **H didn't stop acting; it acted where it shouldn't.** The hallucinations went down (no invented
IDs, no error spirals, half the loops) and over-action came back, mostly in airline: the failure
SFT v2 was built to fix (SEL-71: unwarranted airline cancellations 14 → 2). The Tool / Arg / Loop
rewards check *how* a call is made, not *whether* it should be made: a cancellation with a real
reservation ID passes all three, and Chiara's classifier doesn't flag a wrong decision either.
GRPO pushed towards confident, valid calls, and the only brake on a wrong one was the single-turn
`fix` reward. This matches SEL-73: v2's remaining failures are wrong decisions, not hallucinations.
A heal reward needs a decision check (for example the expected-action matching SEL-73 uses for its
fix points), not only a structural one. Small numbers (airline has 6 tasks) and telecom's extra
failures aren't covered by this diff, so read it as a direction.

**Reading it.** The heal loop did what the rewards asked on single turns (held-out reward up on all
three types, Tool failures gone in training, Arg halved) without G's collapse, and it carried over to
whole conversations as *less looping*, not more success. v2 already made few Tool/Arg errors (5/72 on
dev), so there was little of the targeted failure left to remove, and the Arg fixes taught asking
rather than solving. Next, if there's time: train from base Aya (P, which fails by Tool/Arg 52/60),
where the heal targets are common, or score whole conversations rather than single turns.

**Not done:** the 4 crashed telecom validation conversations (they can't change the decision), H's
Tool/Arg count on validation, and the paired comparison with Arm A (needs Charlie's and Chiara's
`A-*-dev.jsonl`). All three run without a GPU except the first.

