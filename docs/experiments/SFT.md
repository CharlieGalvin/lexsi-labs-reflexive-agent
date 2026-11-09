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
- v1 (S): retail 6, 6 /13; airline 3, 1 /5; telecom 6, 3 of 8 scored (8 crashed, to re-run).
- On these unseen tasks, **v1 equalled the base model on airline** (2/5 on average each), confirming its airline weakness without using dev.
- v2 on validation is still to run: it's the dev-independent test of the airline fix.

## Version log

| Version | Date | Data | Change from the previous version | Dev success (mean of 2 samples) | Tool/Arg | Gate against P | Adapter |
|---|---|---|---|---|---|---|---|
| v1 | 8 Nov | 112 healable fixes (golden v1.1) | — | 13.5/36 (38%) | 0% | Reject (airline −2) | `reagent-lexsi/sft-aya32b-r16` |
| v2 | 8 Nov | v1 + 34 extra airline passes (airline 19 → 53) | Data only (airline); 74 steps against 56 | **21.5/36 (60%)** | 4.2% | **Promote** (airline +1) | `reagent-lexsi/sft-aya32b-r16-v2` |

## Next idea, beyond v2

v2 adds *off-policy* data: teacher conversations, which never reach the states where S goes wrong. On-policy corrections are the standard fix in imitation learning (DAgger): run S on train tasks, find its first wrong action, and have the teacher continue from that point (#50's `PrefixTeacher`). That's the project's heal loop with this adapter as the student, and it's to discuss with Chiara (SEL-68).
