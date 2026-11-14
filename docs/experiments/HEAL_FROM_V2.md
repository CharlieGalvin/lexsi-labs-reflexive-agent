# A heal cycle from SFT v2 (SEL-68 follow-up)

**Status:** the 13 Nov plan below (healing v2 turn by turn) was **not run**: its diagnosis was checked
on 14 Nov and showed it would find little to fix. It's replaced by a **trajectory-level cycle from the
base model**, fixed 14 Nov before any run ([below](#amended-14-nov-before-any-run-a-trajectory-level-cycle)).
**Owner:** Charlie. Code: SEL-68 (#61), plus the validation exclusion and `--fix-mode trajectory` on
`charliegalvin/sel-68-heal-from-v2`.

## Question

The first cycle ([HEAL_FIRST_CYCLE.md](HEAL_FIRST_CYCLE.md)) healed the **base model**: 71% fewer
Tool / Arg / Loop findings, but 4/36 dev tasks before and after, so the gate rejected it. Does the
same loop improve our **best** student, SFT v2?

## What the loop can see (diagnosis, no GPU, from saved rollouts)

The loop only fixes failures where the classifier (`reflexive.hallucination`) finds a Tool, Arg or
Loop mistake. On the saved rollouts, the first finding of each failed conversation:

| Model, set | Failed | First finding | Where |
|---|---:|---|---|
| Base (P), dev | 64 | ARG 55, TOOL 7, NONE 2 | 58 at turn ≤ 3 |
| v2 (S2), dev | 29 | **NONE 20**, LOOP 8, ARG 1 | later turns |
| v2, validation | 26 | **NONE 17**, LOOP 7, TOOL 2 | later turns |
| v2, train pool | 57 | NONE 24, LOOP 27, ARG 3, TOOL 3 | later turns |

What v2's failures are (`just arm-why S2`, re-scored, judges off):

- **retail**: missed state changes. The expected `modify_pending_order_items`,
  `exchange_delivered_order_items`, `return_delivered_order_items`, `cancel_pending_order` or
  `modify_pending_order_address` never happens (it stops short or decides wrongly).
- **airline**: missed `cancel_reservation` on dev; on validation no missed action at all, so the
  failures are in what it says (NL / communicate checks, judged).
- **telecom**: device states not reached (MMS, mobile data, speed) and missed `enable_roaming` or
  `transfer_to_human_agents`.

So about two thirds of v2's failures have **no classifier finding**: they're decisions, not
hallucinations, and the loop as built can't pick them. What it can pick is mostly loops, which
GRPO arm H already showed don't move success on their own. **Prediction: few fixes, mostly LOOP,
and a gate rejection.** The run tests that prediction on the loop as built.

## Setup

| | |
|---|---|
| Student | SFT v2 (`sft2` on vLLM), `configs/sft/arms.pod_sft2.yaml`, arm S2 |
| Heal tasks | `--sample 20` per domain, seed 0, **validation tasks left out** (59 tasks: retail 20, airline 19, telecom 20). The first cycle's sample included 9 validation tasks |
| Teacher, fix check | as the first cycle (GLM-5.3-Flash, `heal.PrefixTeacher`, 3 attempts) |
| Training | `configs/train/sft_heal.yaml`, continuing v2 (`HEAL_INIT_ADAPTER=runs/sft-aya32b-r16-v2`) |
| Gate | `DevGate`, `configs/gate.yaml`: v2 against the healed adapter on all 36 dev tasks |

## Decision rule (fixed before the run)

1. **Primary:** the agreed gate, unchanged: net gain ≥ +2 dev tasks, Tool/Arg/Loop findings not up
   by more than 10%, no domain down 2 or more. Pass = the loop improves v2.
2. **If it passes**, and only then: the healed adapter on the 30 validation tasks × 2, against v2's
   17.0, by the SFT rule (≥ 19.5, no domain more than 1 below v2). A dev pass alone is a hint.
3. **Also recorded either way:** fixes accepted by failure type, and the share of failures with no
   turn to fix. A rejection is a valid result.

This is the gate's second decision (`max_cycles: 3`), and the ninth model compared with v2 on
these tasks: read a pass with that in mind.

## Amended 14 Nov, before any run: a trajectory-level cycle

### Why the 13 Nov plan wasn't run

Its diagnosis, recomputed on 14 Nov from the saved runs with the classifier, matches the table above
exactly (dev 29 failed: no finding 20, LOOP 8, ARG 1; validation 26: 17, 7, 2; train pool 57: 24, 27,
6). A simulation of its heal set (`--sample 20`, seed 0, validation left out, v2's train-pool runs)
gives about 36 failures, 21 with a turn to fix, and **at most 16 training fixes** after the cap of 2 per
lesson, mostly LOOP: the first cycle's size. And at v2's level the gate can't tell a small effect from
noise: v2's own two dev samples were 24 and 19 of 36, against a margin of +2 on one sample.

### What changes, and why

The first cycle trained on **one turn** per failure: 938 target tokens, about 1% of the conversations.
SFT v2, which the same gate promoted (`gate-check P S2`: dev 60% against the base model's 11%), trained on **every
agent turn** of the teacher's verified solutions. The difference between the two is the grain of the
fix, not the loop. So the runtime loop now uses what the offline experiments taught:

| | First cycle | This cycle |
|---|---|---|
| Student | base (P) | base (P), `configs/sft/arms.pod.yaml` |
| Heal tasks | `--sample 10` | **all 112 train tasks outside validation** (no `--sample`) |
| Fix | the teacher rewrites the failing turn | **`--fix-mode trajectory`**: the task's whole verified teacher conversations, from SFT v2's data (`data/train/sft-v2.jsonl`, groups `healable` and `extra`: the golden trajectories plus the extra airline passes) |
| Dedupe | 2 fixes per lesson | none: each conversation is its own example |
| Training | `sft_heal.yaml` | **`sft_heal_trajectory.yaml`**: SFT v2's settings (rank 16, 2 epochs, lr 1e-4, batch 1 × 4), a new adapter from the base model |
| Gate | DevGate, 36 dev tasks | the same, unchanged: its **second** decision (`max_cycles: 3`) |

Only failures are fixed: a task the base model passes in this run gets no conversation, so the cycle
trains on at most v2's 146 examples.

### Decision rule (fixed before the run)

1. **The agreed gate, unchanged**: net gain ≥ +2 dev tasks, Tool/Arg/Loop findings not up by more
   than 10%, no domain down 2 or more. Promote = the automated loop reproduces the offline result,
   and the adapter is swapped in.
2. **Reported whichever way it goes**, with the fixes by domain and the training examples.
3. No validation or test run afterwards: the data is v2's, and v2 is already scored on test (51.5%).

### What a pass does and doesn't show

**Expected: promote**, since v2's data already passed this gate against the base model. This is a
**demonstration** that the runtime pipeline (detect → fix → retrain → gate → swap) can do automatically
what the offline experiments did by hand. It is **not new evidence** that healing works: dev shaped
v2's data (v1's airline rejection led to the airline passes), so a pass on dev is expected, and the
held-out claim stays v2's test result. The comparison it adds is the grain of the fix: one turn
(findings −71%, success unchanged, rejected) against whole conversations.

