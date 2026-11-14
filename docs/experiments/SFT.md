# SFT on the verified teacher fixes: v1 (done) and v2 (in progress)

**Status (8 Nov, late):**
- **v1 is done: indicative, dev only.**
- **v2 is done on dev and meets every condition of its decision rule, fixed before it ran.** Its validation run, the dev-independent check, is still to come.

**Tickets and code:**
- Tickets: [SEL-65](https://linear.app/self-healing-hallucination/issue/SEL-65) (v1) and [SEL-71](https://linear.app/self-healing-hallucination/issue/SEL-71) (v2).
- Code: PR #49 for training and serving, PR #53 for `just arm-eval`.
- Adapter: `reagent-lexsi/sft-aya32b-r16` (v1, private).
- How to run it on a GPU: [`SFT_POD.md`](SFT_POD.md).

**The question:** does plain supervised fine-tuning (SFT) on the teacher's verified fixes heal Aya Expanse 32B? This is the simplest form of the project's heal: verified teacher fixes, retrain, gate. It's a side experiment alongside Katie's GRPO (SEL-49) and doesn't replace it.

## What each split is for

| Split | Job |
|---|---|
| **Train, trained on** (112 tasks in v1) | Diagnosis only (underfitting?). Never evidence of improvement. |
| **Train, never trained on** (30: 13 retail, 5 airline, 12 telecom) | **Validation**: guides decisions. That's the 15 held-out `solved_by_aya` tasks and the 15 without a teacher fix. |
| **Dev** (36) | **The gate**: once per version, judged by a rule fixed in advance. v1's dev failures were read to design v2, so v2's dev numbers are development evidence, not a claim. |
| **Test** (100) | Never run on these adapters. Only a model the team declares as an arm runs on test, once. |

## Shared setup (every version)

| | |
|---|---|
| **Base model** | `CohereLabs/aya-expanse-32b` |
| **Method** | QLoRA: 4-bit NF4 base, LoRA rank 16 / alpha 32 on all linear layers, lr 1e-4 cosine, 2 epochs, batch 1 × 4 |
| **Loss** | Only on the agent's turns (`train_on`), computed on those positions alone (`target_loss`, `logits_to_keep`) |
| **Held-out loss** | The 15 `solved_by_aya` fixes, never trained on |
| **Serving** | vLLM 0.33 with the adapter as `sft`, the relaxed chat template, 32K context (`scripts/pod/serve.sh`) |
| **Arms** | **P** = the base model on the same server (`configs/sft/arms.base.yaml`); **S** = base + adapter (`configs/sft/arms.sft.yaml`). P is the fair control: only the adapter differs. |
| **Scoring** | `just oracle-score` (AuditKIT oracle, scoring v3, judges on), `just arm-metrics`, `just arm-eval`, `just arm-failures`, `just arm-samples`, `just gate-check` |

## Reproduce v1

On the pod (details and fixes in [`SFT_POD.md`](SFT_POD.md)):

```bash
bash scripts/pod/setup.sh                               # model, golden v1.1, both training-data files
just sft-train --dry-run                                # 112 train / 15 held out, ~1.04M tokens
just sft-train --push                                   # configs/train/sft_aya32b.yaml → runs/sft-aya32b-r16/
just sft-serve                                          # server on :8000
for a in S P; do cfg=$([ $a = S ] && echo configs/sft/arms.sft.yaml || echo configs/sft/arms.base.yaml)
  for d in retail airline telecom; do just rollout $a $d --split dev --samples 2 --config $cfg; done; done
```

On the Mac, after copying `results/episodes/arms/{P,S}-*-dev.jsonl` across:

```bash
just oracle-score results/episodes/arms/S-*-dev.jsonl   # and P
just arm-samples P S                                    # spread between samples
just arm-failures results/episodes/arms/S-*-dev.jsonl --out results/S-dev-failures.md
just gate-check P S                                     # the agreed gate, as a check
```

## v1 results (8 Nov)

**Training:**
- 56 steps, 23 min on one H100 SXM, peak 54 GB.
- 95.7M trainable parameters (0.30%).
- Train loss 0.450. Held-out loss 0.854 → 0.393: it fell, then flattened, and never rose, so no sign of memorising.

**Success on dev:**

| | Retail | Airline | Telecom | Total |
|---|---|---|---|---|
| Arm A (Cohere API) | 1/15 | 1/6 | 0/15 | 2/36 |
| P, sample 0 | 2/15 | 3/6 | 0/15 | 5/36 |
| P, sample 1 | 1/15 | 1/6 | 1/15 | 3/36 |
| **S, sample 0** | 8/15 | 1/6 | 7/15 | **16/36** |
| **S, sample 1** | 5/15 | 1/6 | 5/15 | **11/36** |

- **Mean of the two samples: S 13.5/36 (38%) against P 4/36 (11%).** Pooled, 27/72 against 8/72.
- **Sample 0, paired by task:** 14 passed only with the adapter, 3 only without (exact McNemar p = 0.013).
- **The oracle confirms sample 0** (S 16, P 5). Sample 1 is the runs' own verdicts.
- **Run-to-run spread is about 5 tasks out of 36 for S.** 16/36 was the better of two draws.

**Hallucination**, as the share of conversations with a Tool or Arg hallucination (sample 0):

| | Retail | Airline | Telecom |
|---|---|---|---|
| A | 53% | 67% | 100% |
| P | 67% | 33% | 87% |
| **S** | **0%** | **0%** | **0%** |

- The gate's own count, findings per conversation, falls from **8.56 → 0.94**.
- S makes *more* tool calls than P (16.1 against 9 per telecom conversation), and all of them are valid.

**The gate (`just gate-check P S`, a check, not recorded): REJECT, decided by the domains check.**
- Pass rate: ✓ net +11.
- Hallucination: ✓.
- Domains: ✗ **airline −2** (retail +6, telecom +7).
- P's airline 3/6 was its luckier draw (sample 1: 1/6). With 6 tasks and one sample, the domains rule is noisy. That's a reason to propose `samples_per_task: 2` for *future* gates, not to re-gate v1 on another sample.

**Failure analysis (`just arm-failures`, sample 0):**
- **None of S's 20 failures are hallucinations or communication failures.** 11 are a wrong database end state, 5 are telecom phones left unfixed, and 4 are step-limit loops.
- **Airline: eligibility decisions.** S cancelled reservations the policy doesn't allow (tasks 39, 41, 43) and sent an unwarranted certificate (task 36).
- **Telecom: repeats.** S refuels the same line again and again when the phone still doesn't work.
- **Restraint examples aren't missing from the data.** 8 of the 19 airline training examples are hold-back cases, and the teacher explains its eligibility check in 19 of 23 airline conversations.

**Validation set:** running. S so far: retail 12/26, airline 4/10 (runs' own verdicts).

**Caveats:**
- One or two samples per task, on 36 tasks.
- Three crashed S conversations in sample 0 were re-run as new conversations; one passed.
- Dev's Arm A baseline (6%) is below test's (15%).
- The judges may favour the teacher's style, which S now copies.
- P and S ran at 32K context, against Arm A's 128K.

## v2 (SEL-71): more varied airline examples. Plan fixed before any result.

**Hypothesis:** v1 underfits airline's eligibility rule (19 examples). Adding the teacher's attempts 2–3 at temperature 0.7 on the same airline **train** tasks, each verified by the oracle, should cut ineligible cancellations without losses elsewhere.

**One change: the data.**
- `just sft-v2-data` (`scripts/data/build_extra_fixes.py --domains airline`) keeps only clean passes on train tasks in v1's `healable` group. It leaves out any attempt v1 already uses and any word-for-word repeat.
- Config `configs/train/sft_aya32b_v2.yaml` differs from v1's only in `name`, `data` and `train_group`.

**Known confound:** more examples also means more optimiser steps (about 75 against 56).

**Decision rule.** v2 is better than v1 if all of these hold:
- `just gate-check P S2` promotes it;
- airline improves by at least 2 tasks against v1;
- retail and telecom each lose at most 1;
- Tool/Arg hallucination stays at or below 5% of conversations;
- the overall change exceeds v1's run-to-run spread (about 5 tasks).

**Stop rule:** if over-action doesn't fall by a third, or success drops, report it and stop. Budget about $25.

**Reproduce v2:**

```bash
# Mac (teacher via OpenRouter, judges via OpenAI; about $1):
python scripts/teacher/fixes.py --domains airline --tasks all --all-attempts
# Pod, with results/fixes/glm53f.jsonl copied up:
just sft-v2-data                                        # → data/train/sft-v2.jsonl + .report.json
just sft-train configs/train/sft_aya32b_v2.yaml --dry-run
just sft-train configs/train/sft_aya32b_v2.yaml --push
EXTRA_ADAPTERS="sft2=runs/sft-aya32b-r16-v2" just sft-serve   # v1 as "sft", v2 as "sft2", one server
for d in retail airline telecom; do
  just rollout S2 $d --split dev --samples 2 --config configs/sft/arms.sft2.yaml   # → S2-<domain>-dev.jsonl
done
# Mac: just oracle-score …S2-*-dev.jsonl; just arm-samples S S2 P; just gate-check P S2
```

**v2 results (8 Nov):**

**Training:**
- 146 conversations: v1's 112 plus 34 extra airline passes from 18 train tasks, so airline goes from 19 to 53.
- 74 steps, 28 min.
- Train loss 0.426. Held-out loss → **0.368**, levelling off without rising (v1: 0.393).
- Adapter: `reagent-lexsi/sft-aya32b-r16-v2`.

**Success on dev, two samples per task** (the runs' own verdicts; the gate re-scores sample 0 with the oracle and agrees, 24/36):

| | Retail | Airline | Telecom | Total, per sample | Mean |
|---|---|---|---|---|---|
| P (base) | 2, 1 /15 | 3, 1 /6 | 0, 1 /15 | 5, 3 /36 | 4.0 |
| S (v1) | 8, 5 /15 | 1, 1 /6 | 7, 5 /15 | 16, 11 /36 | 13.5 |
| **S2 (v2)** | **8, 8 /15** | **4, 3 /6** | **12, 8 /15** | **24, 19 /36** | **21.5** |

**The gate (`just gate-check P S2`, a check, not recorded): PROMOTE.**
- Net gain +19 tasks: 21 passed only with v2, 2 only with P (McNemar p = 0.0001).
- Domains: airline +1, retail +6, telecom +12.
- Findings per conversation: 8.56 → 1.17.

**The decision rule:**

| Condition | Result | |
|---|---|---|
| The gate promotes v2 over P | Promote | ✓ |
| Airline +2 or more against v1 | +2.5 per sample | ✓ |
| Retail and telecom lose at most 1 | +1.5 and +4 per sample | ✓ |
| Tool/Arg at or below 5% of conversations | 3/72 = 4.2% | ✓, just |
| Overall change larger than v1's spread (about 5) | +8 per sample | ✓ |

**Caveats:**
- v2 was designed after reading v1's dev failures, so these are development numbers.
- Known confound: 74 steps against 56. Telecom improved by +4 per sample with no new telecom data.
- Tool/Arg crept up from v1's 0/36 to 3/72 (2 in retail).
- Airline is 6 tasks.

**Validation** (the 30 unseen train tasks, 2 samples, runs' own verdicts):
- P: 9.0/30 on average.
- v1 (S), before its telecom crashes were re-run: retail 6, 6 /13; airline 3, 1 /5; telecom 6, 3 of 8 scored (8 crashed). Superseded by the table in the next section (telecom 7 and 5 of 12, all scored).
- On these unseen tasks, **v1 equalled the base model on airline** (2/5 on average each), confirming its airline weakness without using dev.
- v2 on validation is still to run: it's the dev-independent test of the airline fix (done 9 Nov: see the next section).

## Validation (9 Nov): v2 on tasks no arm trained on

The dev-independent check: the 30 validation tasks (13 retail, 5 airline, 12 telecom), 2 samples
each, scored in-run (`just arm-samples P S S2 --split val`).

| Arm | Retail | Airline | Telecom | Total, mean of 2 | Samples |
|---|---:|---:|---:|---:|---:|
| P (base) | 5.0 | 2.0 | 2.0 | **9.0** | 8, 10 |
| S (v1) | 6.0 | 2.0 | 6.0 | **14.0** | 16, 12 |
| S2 (v2) | 6.5 | 2.5 | 8.0 | **17.0** | 16, 18 |

- **SFT generalises:** v2 scores +8 over the same base model on unseen tasks, well above either
  arm's spread between samples.
- **v2 against v1 is +3, within noise** (v1's own samples differ by 4), and no domain is worse.
- **The airline hypothesis isn't supported here:** airline is 2.5 against 2.0. The gain is in
  telecom, where v2 got no new data. So v2's extra steps (74 against 56) may explain it: the
  known confound. Until SL is in, credit v2's gain over v1 to "more training and/or more data".
- On dev, v2 did fix airline's over-action (unwarranted cancellations 14 → 2, `just arm-failures`),
  but dev informed v2's design, and that gain doesn't show detectably on 5 unseen airline tasks.

Commands: the task list for each domain comes from `P-<domain>-val.jsonl`. A resume rewrites an
arm's own file without its crashed records, so a task whose samples all crashed vanishes from it.
Pass `--out`: the validation tasks are train tasks, so without it the file is named
`<arm>-<domain>-train.jsonl`, which `--split val` doesn't read.

```bash
# Pod, one arm at a time (two at once crashed telecom); <arm> and its config, e.g. S2 configs/sft/arms.sft2.yaml
for d in retail airline telecom; do
  ids=$(jq -r '.task_id' results/episodes/arms/P-$d-val.jsonl | sort -u | tr '\n' ' ')
  just rollout <arm> $d --task-ids $ids --samples 2 --config <config> \
    --out results/episodes/arms/<arm>-$d-val.jsonl
done
# Mac: just arm-samples P S S2 SL S3 --split val
```

## SL and v3 (SEL-72): plan fixed before any result

Three arms at the same **74 steps**. They differ in data, and so in passes over it: at 4 examples
a step, SL makes about 2.6 passes over its 112, v2 2.0 over its 146 and v3 1.8 over its 166 (v3
sees about 40 examples only once, which ones set by the shuffle). Matching steps keeps the compute
equal, not the epochs:

| Arm | Data | Config |
|---|---|---|
| SL | v1's 112 | `configs/train/sft_aya32b_long.yaml` |
| S2 (v2) | v1's 112 and 34 airline | `configs/train/sft_aya32b_v2.yaml` |
| S3 (v3) | v1's 112, 34 airline and 20 retail (166) | `configs/train/sft_aya32b_v3.yaml` (`just sft-v3-data`) |

Telecom has no extra verified passes, so it stays at v1's 47. Tests check that SL differs from v1
only in its steps, and v3 from v2 only in its data and steps.

**Reproduce v3:**

```bash
# Mac (teacher via OpenRouter, judges via OpenAI): v3's retail extras need the retail fix run too
python scripts/teacher/fixes.py --domains retail airline --tasks all --all-attempts
# Pod, with results/fixes/glm53f.jsonl copied up:
just sft-v3-data                                        # → data/train/sft-v3.jsonl: 166 (146 without the retail run)
just sft-train configs/train/sft_aya32b_v3.yaml --dry-run
just sft-train configs/train/sft_aya32b_v3.yaml --push
EXTRA_ADAPTERS="sft2=runs/sft-aya32b-r16-v2 sftlong=runs/sft-aya32b-r16-long sft3=runs/sft-aya32b-r16-v3" just sft-serve
# then the validation rollouts above, with S3 and configs/sft/arms.sft3.yaml
```

**Reading SL:** SL ≈ v2 means the steps explain v2's gain over v1; SL ≈ v1 means the airline data
helped, just not where expected.

**Decision rule for v3** (on validation, 2 samples each): v3 is better than v2 if its total beats
17.0 by more than v2's sample spread (2), so 19.5 or more; no domain drops by more than 1 task;
and Tool/Arg hallucination stays at or below 5% of conversations. Otherwise it's reported, not
adopted.

**Caveat:** each version compared on the same 30 tasks tunes us to them a little, as dev was. The
headline arm is declared before test (SEL-26), and every test run is reported.

## SL and v3: results (9 Nov)

All five arms on the 30 validation tasks, 2 samples each (`just arm-samples P S S2 SL S3 --split val`;
SL's and S3's telecom crashes came from running both arms at once and were re-run one arm at a time):

| Arm | Data | Steps | Retail (13) | Airline (5) | Telecom (12) | Total, mean | Samples |
|---|---|---:|---:|---:|---:|---:|---:|
| P | — | — | 5.0 | 2.0 | 2.0 | **9.0** | 8, 10 |
| S (v1) | 112 | 56 | 6.0 | 2.0 | 6.0 | **14.0** | 16, 12 |
| SL | 112 | 74 | 6.5 | 2.0 | 7.0 | **15.5** | 15, 16 |
| S2 (v2) | + 34 airline | 74 | 6.5 | 2.5 | 8.0 | **17.0** | 16, 18 |
| S3 (v3) | + 34 airline + 20 retail | 74 | **8.0** | 2.5 | 6.5 | **17.0** | 16, 18 |

**v3 isn't adopted:** its total (17.0) needed 19.5 or more, and telecom fell 1.5 against v2 (at most 1
allowed). v2 stays Arm B and GRPO's starting adapter, as fixed at 19:24 in SEL-72.

**These are the runs' own verdicts** (`just arm-samples`), not re-scored with `just oracle-score`,
which `samples.py` says the reported numbers need; crashed conversations are counted apart, not as
failures, and all were re-run. The telecom condition sits within 0.5 of its limit, so a re-score
could change it, but the decision doesn't rest on it: the total misses 19.5 by 2.5. Re-score before
these numbers go in the report.

**Confound:** at a fixed 74 steps the arms also differ in passes over their data (above), so SL against
v2 against v3, and v3's telecom drop, are data *and* epochs, not data alone.

- **SL sits halfway between v1 and v2** (14.0 → 15.5 → 17.0). Each step of 1.5 is within v1's sample
  spread of 4, so the shares due to training longer and to the airline data can't be separated.
- **v3's retail data helped retail** (8.0, the best of any arm) and **telecom fell** (6.5): at a fixed
  74 steps, more non-telecom examples mean fewer passes over each telecom example.
- **Held-out loss doesn't separate them:** SL 0.365, v2 0.368, v3 0.368 at the end of training.
- **The rule's third part, Tool/Arg at or below 5%:** v3 has 3 of 60 conversations (5.0%), so it
  holds; v3 fails on the first two parts. For reference: SL 4 of 60, v2 4 of 60 (6.7%, all telecom's
  customer-only phone tools), v1 6 of 60, P 52 of 60 (`classify_record`, any Tool or Arg finding).

**Failure analysis** (one main cause per failed conversation, validation, both samples):

| Main cause | v1 | v2 | v3 |
|---|---:|---:|---:|
| Loop / hit the step limit | 13 | 5 | 8 |
| Over-action (an unneeded change) | 5 | 5 | 4 |
| Right action, wrong arguments | 5 | 4 | 9 |
| Missed an action | 2 | 4 | 5 |
| Did something else instead | 3 | 1 | 0 |
| Past 32K context / too many errors | 3 | 4 | 0 |
| Actions right, another check failed | 1 | 3 | 0 |
| **Failed (of 60)** | 32 | 26 | 26 |

v2's gain over v1 is mostly fewer loops (13 → 5). 14 of its 26 failures are wrong decisions at a
particular turn; airline still over-acts on unseen tasks (3 of its 5 failures), so the dev fix
(cancellations 14 → 2) doesn't carry over. 4 conversations ran past 32K context.

**Training** (both pushed to `reagent-lexsi`): SL 31 min, train loss 0.401, held-out 0.854 → 0.365;
v3 29 min, train loss 0.408, held-out 0.854 → 0.368. Neither held-out loss rose.

## v4 (SEL-73): on-policy fixes at v2's own wrong decisions

Plan and rule fixed at 21:16 on 9 Nov, before any result: v4 replaces v2 if its validation total is
19.5 or more, no domain drops by more than 1, and Tool/Arg hallucination stays at or below 5%.

**Method** (`just onpolicy-fixes`, `scripts/heal/onpolicy_fixes.py`):

1. v2 on the 112 train tasks outside validation, 1 sample (`S2-<domain>-trainpool.jsonl`): **55 passed**
   (retail 21/46, airline 8/19, telecom 26/47). Most are tasks v2 trained on: it underfits.
2. The turn to fix, from the task's expected actions: the first change that matches none still pending;
   else the last agent turn if an expected change was never made; else the classifier's first finding.
3. GLM-5.3-Flash rewrites that turn (`heal.fix_point`, `check_fix`, `to_sft_example`), up to 3 attempts;
   accepted only if every change in it is one the task still expects.

| Fix point | Found | Accepted |
|---|---:|---:|
| Unexpected change | 38 | 26 |
| Classifier (loop / hallucination) | 3 | 3 |
| Change never made | 13 | 0 |
| No fix point (actions matched) | 3 | — |

- **No stop-short fixes:** at those points the teacher looked something up or asked to confirm (which
  the policy requires), but the rule demanded the change itself in that turn. **Also a bug, fixed
  after this run (review of #56):** at a change-never-made point the script counted the last turn's
  own changes as already made, though that turn is the one rewritten, so a correct rewrite that
  redid them was refused. Calls the environment rejected ("Error…") also counted as changes. Both
  are fixed; a re-run would find different fix points and accept different fixes, so the 29 of 54
  and v4's data are from the earlier version.
- **OpenRouter credit ran out** at fix point 51: the last 3 were refused for that reason.
- **A spot check of 8** found the accepted turns sound: they confirm, re-check or hold back instead of
  v2's premature or wrong change.

**Data and training:** `just sft-v4-data` (v2's file + 29 on-policy, group `onpolicy`); trains on 175
(112 + 34 + 29) for 74 steps; held-out loss tracked SL's and v3's.

**Result** (validation, 2 samples each, the runs' own verdicts): retail **7.0**, airline **3.5** (best
of any arm), telecom 6.5, total **17.0** (samples 18, 16). **Not adopted:** total below 19.5, telecom
−1.5 against v2. The rule's third part, Tool/Arg at or below 5%, isn't computed yet
(`classify_record` on `S4-*-val.jsonl`); the decision holds whatever it is. The
fixes moved the domains they target, and telecom fell as it did for v3: at a fixed 74 steps, more
non-telecom examples mean fewer passes over telecom's (about 2.0 per example for v2, 1.7 for v4).

**Next, after submission:** v4's data at about 90 steps (restore telecom's passes); a stop-short rule
that accepts a look-up or confirmation and checks the following turns.

## Version log

| Version | Date | Data | Change from the previous version | Dev success (mean of 2 samples) | Validation, mean of 2 (of 30) | Tool/Arg | Gate against P | Adapter |
|---|---|---|---|---|---|---|---|---|
| v1 | 8 Nov | 112 healable fixes (golden v1.1) | — | 13.5/36 (38%) | 14.0 | 0% | Reject (airline −2) | `reagent-lexsi/sft-aya32b-r16` |
| v2 | 8 Nov | v1 + 34 extra airline passes (airline 19 → 53) | Data only (airline); 74 steps against 56 | **21.5/36 (60%)** | **17.0** | 4.2% | **Promote** (airline +1) | `reagent-lexsi/sft-aya32b-r16-v2` |
| SL | 9 Nov | v1's 112 | Steps only: 74 (the ablation for v2's confound) | — | 15.5 | | | `reagent-lexsi/sft-aya32b-r16-long` |
| v3 | 9 Nov | v2 + 20 extra retail passes (retail 46 → 66) | Data only (retail), at 74 steps (SEL-72) | — | 17.0 (not adopted) | | | `reagent-lexsi/sft-aya32b-r16-v3` |
| v4 | 9 Nov | v2 + 29 on-policy teacher fixes (retail, airline, telecom) | Data only (on-policy), at 74 steps (SEL-73) | — | 17.0 (not adopted) | | | `reagent-lexsi/sft-aya32b-r16-v4` |

## Next ideas

v2's teacher data is off-policy; v3 added more of it and v4 added on-policy fixes (DAgger, above).
Neither beat v2 on validation. After submission: v4's data at about 90 steps (restoring telecom's
passes), and a stop-short rule that accepts a look-up or a confirmation and checks the turns that
follow it. GRPO from v2 (SEL-21, SEL-49) is the next step for the project itself.
