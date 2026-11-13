# A heal cycle from SFT v2 (SEL-68 follow-up)

**Status:** plan and rule fixed 13 Nov, before the run. Results to follow in this page.
**Owner:** Charlie. Code: SEL-68 (#61) plus the validation exclusion on `charliegalvin/sel-68-heal-from-v2`.

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
