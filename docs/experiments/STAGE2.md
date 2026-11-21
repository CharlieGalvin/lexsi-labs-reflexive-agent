# Stage 2 pre-registration

Every rule here is written before the run it judges, as in Stage 1. Agreed by Charlie, Katie and
Chiara on Fri 20 Nov 2026 and committed on Sat 21 Nov, before any Stage 2 run except the two listed
under "Launched before this file". Tickets: SEL-79 and the SEL-80 to SEL-113 range in Linear (milestone
M4). Deadline: Tue 24 Nov, 23:59 AoE (Wed 25 Nov, 12:00 UK); we aim to submit Tue evening.

## The question

Stage 1 healed made-up tools and values: on τ²-bench test, SFT v2 cut conversations with a Tool or
Arg hallucination from 95.0% to 10.0% and raised task success from 14.5% to 51.5%. In 37 of v2's
55 failed dev and validation conversations the classifier finds no hallucination at all: what is
left is mostly decisions. Stage 2 asks:

1. Can a checker that reads only the conversation and the policy (no answer key) find **made-up
   intent**: a change the customer never asked for, or made before they said yes?
2. Does stopping those changes live (the **guard**) solve more tasks than v2?
3. Can v2 learn the corrected behaviour, so it no longer needs the guard?
4. Did healing cost Aya anything elsewhere?

## Constants for every Stage 2 run

- Customer: GPT-4.1, pinned snapshot `gpt-4.1-2025-04-14`, through OpenAI, as in Stage 1. A run
  with any other customer is not comparable and is reported as such.
- Harness, prompt formats, decoding (temperature 0), splits (`configs/splits.json`) and τ²-bench
  scoring as in Stage 1. Every success re-scored by AuditKIT's oracle, judges on.
- Self-hosted arms: one vLLM server, 32K context, 8 workers, as Arm B.
- The loop monitor (`no_progress`, SEL-69) stays in `log` mode for every Stage 2 arm, as in Stage 1:
  it records loops but never nudges or stops. Only the action guard below acts, and only where an
  arm says so.
- Budget: $200 for GPU and API spend in total. The $200 Hugging Face credits are reported as
  already used (21 Nov; raised with the organisers); if they are restored they are spent first. Stop
  and agree as a team before going past the $200.
- Keys: the OpenAI key for the pinned customer and τ²-bench's judges is the one already set in the
  RunPod environment variables; runs that need it run on the pod. Keys are never committed.

## What each split is for

| Split | Size | Its one job in Stage 2 |
|---|---|---|
| Train | 112 tasks | Building and tuning the checker; collecting the guard's fixes; training data |
| Validation | 30 tasks × 2 | The guard decision (Decision 1), and the retrained model's first score |
| Dev | 36 tasks | The gate's third and last decision (Decision 2) |
| Test | 100 tasks × 2 | One declared run per arm below, reported whatever it shows |

Nobody opens a Stage 2 test conversation until every Stage 2 test number is in. The 37 and the
classifier audit use Stage 1 dev and validation conversations only, never test.

## The checker: made-up intent

A fourth finding type beside Tool, Arg and Loop, behind one function,
`check_action(conversation_so_far, proposed_call, policy) -> finding | None` (SEL-83). Every rule
reads only the conversation and the policy, never τ²-bench's expected actions; a test enforces this.
Every rule is marked **live** (checkable at the moment of a change, fast; the guard may use it) or
**after the fact** (gate and reporting only).

**The "no yes" rule (SEL-88), hand-written, live.** A call to one of the agent's state-changing
tools, with no explicit yes from the customer since the agent described that specific change. An
explicit yes is a clear agreement in the customer's turn after the description ("yes", "go ahead",
"please do", "that's fine", "confirm", and equivalents). The description says what changes: the
action and the order, reservation or line it applies to. A request up front ("cancel it please") is
not a yes to the agent's later proposal. One yes to a list of changes the agent set out together
covers every change on that list; a yes to an earlier proposal doesn't cover a different change made
later. The same definition is used for the `no_yes` category when labelling the 37
(`docs/experiments/LABELLING_GUIDE.md`). State-changing tools are τ²-bench's own `WRITE` tools, read
from the live environment; today they are: retail `cancel_pending_order`,
`exchange_delivered_order_items`, `modify_pending_order_address`, `modify_pending_order_items`,
`modify_pending_order_payment`, `modify_user_address`, `return_delivered_order_items`; airline
`book_reservation`, `cancel_reservation`, `send_certificate`, `update_reservation_baggages`,
`update_reservation_flights`, `update_reservation_passengers`; telecom `disable_roaming`,
`enable_roaming`, `refuel_data`, `resume_line`, `send_payment_request`, `suspend_line`.
Built and tested on v2's train-task conversations only.

**Amended Sat 21 Nov, before any Stage 2 run: a word match, then an LLM judge (SEL-88, PR #74).**
On v2's train-task conversations the word match alone flagged 10 of 55 passes and 11 of 57
failures. Read by hand, about half of the flagged passes were a yes the words couldn't see: an
instruction after the details were set out ("Just cancel them, please"). So:

- **A clear instruction counts as a yes**, the same as an agreement, when it comes after the change
  or the details it applies to have been set out. A request or question before that, followed
  straight away by the change, still doesn't ("Can I return the backpack too?").
- **Two stages.** The word match runs first and is free. Only a change it would flag goes to an LLM
  judge: `gpt-4.1-mini-2025-04-14` (pinned, not the customer's model), temperature 0, the prompt
  in `reflexive.intent_judge`, answers cached by the exact request. The judge sees agent and
  customer turns and the agent's calls, never the system prompt, tool results or the answer key.
- **The judge must quote the customer.** It answers `yes_given`, `quote` and `reason`; a yes counts
  only if the quote is the customer's own words from one of their messages.
- **A judge error never blocks.** It is logged as an after-the-fact finding, `no_yes_judge_error`,
  and counted in Decision 1's report.
- The word match's train-task flag rates are kept (`--judge off`) beside the two-stage rule's, so
  the paper can say what the judge adds.
- **Telecom: only `refuel_data` is checked.** Retail and airline policies require an explicit yes
  before any change; telecom's asks only for the price of a data top-up to be confirmed. With the
  judge, the rule still flagged 3 telecom train passes, all roaming or resume changes telecom's
  policy allows without a yes. Its other write tools (`disable_roaming`, `enable_roaming`,
  `resume_line`, `send_payment_request`, `suspend_line`) are out of scope for the rule and the
  guard; the list of state-changing tools above is kept as τ²-bench's own.

**Meta-healing (SEL-91).** An LLM reads v2's train-task failures that no current rule explains,
plus some passes, and proposes candidate rules (Python checks or fixed judge prompts). v2's
train-task conversations are split two thirds (propose) and one third (accept or reject),
stratified by domain and pass/fail, seed 0. A candidate is accepted only if, on the held-out third:

- it flags at least 2, and at least 20%, of the held-out unexplained failures;
- it flags at most 1 held-out pass (about 1 in 10);
- it reads no answer key (the test passes);
- a person reads 5 conversations it flags and agrees they are real errors.

At most 10 candidates are scored and at most 3 accepted. If none passes, that is the result.

**Amended Sat 21 Nov, 11:15 UK, after meta-healing round 1 and before round 2 (SEL-91, PR #75).**
The proposer is GPT-4.1 (`gpt-4.1-2025-04-14`); each candidate is a fixed judge prompt run by
`gpt-4.1-mini-2025-04-14` at temperature 0, whose quoted evidence must be in the conversation; a
checker error never flags. Unexplained = no Tool / Arg / Loop and no live "no yes" finding. The
split gives 38 held-out conversations, 5 of them unexplained failures (3 retail, 2 airline, 0
telecom), so the bar is in effect 2 of 5.

- **Round 1** proposed 5 candidates, all telecom; none met the bar (0/5 unexplained failures each,
  1 to 8 of 19 passes flagged). Reported as it is.
- **Round 2** uses the 5 candidates left of the 10. The proposer prompt adds: the rules already
  tried (names and descriptions, not their scores), the failures it is shown by domain, rules
  spread across domains in proportion, and "something never done" errors as whole-conversation
  rules. The change was motivated by round 1's output on the propose set; we also knew, from the
  split, that the held-out third has no unexplained telecom failures. Nothing else changes: same
  split, same bar, same models. There is no round 3. Both rounds are reported.
- A live meta-healing rule re-asks the agent with its own fixed one-sentence message instead of
  the "no yes" message; whole-conversation rules are after the fact only.

**Result, Sat 21 Nov 11:25 UK: no rule accepted.** Round 2's
`retail_modify_pending_order_without_collecting_all_items` met the numeric bar (3/5 held-out
unexplained failures, 1/19 passes; 5/14 already-explained failures) and failed human review
(Charlie): 2 of its 5 flagged conversations show the error it names (retail 103 turn 35, retail 8),
1 is wrong for another reason (retail 107: an item change on an order that isn't pending), and 2
are not errors of this kind (airline 33: an airline booking, where the policy has no such step,
though turn 27 drops a flight the customer kept; retail 29: the customer chose a partial change).
The rule's "or similar" let the checker apply it to exchanges, address changes and airline
bookings. Review used 4 of 5 as the bar; that number was not written here before the review, and
2–3 of 5 fails any majority bar too. The other 9 candidates failed the numbers. All 10, with their
scores, are in `results/meta-heal.metrics.json`. The checker freezes with "no yes" only.

**Amended Sat 21 Nov, 11:50 UK, before it is measured: a hand-written "mismatch" rule (PR #76).**
Meta-healing's round 2 and its misfires pointed at changes that don't match the request (see the
result above); a person wrote the tight version. At 7 listed changes (retail: modify items,
exchange, return, cancel order; airline: update flights, update bags, cancel reservation) the
checker asks whether the change differs from what the customer asked for and agreed to, or from
the agent's own lookups. Checked by full GPT-4.1 (`gpt-4.1-2025-04-14`, temperature 0; amended
11:55, still before measuring: mini read meta-healing's rules too loosely), through the same
machinery as meta-healing's rules (evidence must be in the conversation, errors never block), tool
results shown in full; live, with its own block message. "No yes" keeps gpt-4.1-mini. It joins the frozen checker only if it flags at most 1 in 10 of SFT v2's train-task
passes (`just rule-rates`), and of base Aya's (Arm A) train-task passes when those are run too
(validation tasks dropped). Failures flagged, unexplained failures flagged and an injected-error
test (corrupted changes from passing conversations: caught, and originals flagged) are reported
whatever they show. Measured once; the wording is not tuned on the result.

**Freeze: Sat 21 Nov, 23:59 UK (SEL-95).** The checker is tagged `stage2-checker-frozen`; the rule
list and each rule's live / after-the-fact mark are added below this line in the same commit.
Nothing changes the checker after the tag. If meta-healing isn't ready, the freeze goes ahead with
the "no yes" rule only, and meta-healing's rules are reported offline as a separate result.

## The guard (SEL-89)

`reflexive.monitor.ActionGuard`, PR #73. Before a reply is sent to τ²-bench, every state-changing
call in it goes through `check_action`. It has its own mode, separate from the loop monitor's:

- `off`: nothing is checked. `log`: findings are recorded and every call runs.
- `act`: a live finding blocks the whole reply. τ²-bench never sees it, no τ²-bench step is used,
  and the agent is asked again with one fixed message added to that prompt: "Note from the system,
  not the customer: you haven't got the customer's yes. Describe the change and ask them to
  confirm." The conversation carries on.
- Limits: the same change (the same tool with the same arguments) is blocked at most 2 times; the
  third time it runs and is recorded as `overridden`. After 10 blocks in one conversation nothing
  more is blocked, so every conversation ends. After-the-fact findings are only logged.
- If the checker raises an error, the guard switches itself off for the rest of that conversation
  and the call runs; the error is recorded.

Every decision is recorded per call (`blocked`, `overridden`, `held`, `logged`) in
`rec["monitor"]["actions"]`.

## Decision 1: does the guard help? (SEL-96, validation)

SFT v2 with the guard in act mode, validation 30 tasks × 2, with the frozen checker, against v2's
17.0 / 30. **Go for retraining** only if all three hold:

- success at least 19.0 / 30;
- no domain more than 1 task below v2;
- wrongly blocked good changes in no more than 1 in 10 passing conversations.

Also reported: blocks per conversation, overrides, extra turns, conversations where a checker
error switched the guard off, and how often the customer refused or went off-script.
**No-go:** skip retraining; v2 + guard goes to Decision 2.

## Retraining (SEL-100, SEL-102, SEL-103)

- Fixes: v2 + guard (act mode) on the train tasks. Kept only if τ²-bench passes them, with
  `assert_not_test()` on every task.
- Too few: if fewer than 30 conversations pass, report the count, skip retraining, and v2 + guard
  goes to Decision 2. No top-ups from any other data.
- Recipe: from the base model, exactly SFT v2's recipe (`configs/train/sft_aya32b_v2.yaml`: 2
  epochs, learning rate 1e-4, same QLoRA settings), on v2's data plus the fixes as their own group.
  One change from v2: the added fixes. Adapter published with its `lexsi_provenance.json`.
- First score: validation 30 × 2 **without** the guard, reported whatever it shows.

## Decision 2: the dev gate, third and last decision (SEL-104)

The candidate is the retrained model (without the guard) if retraining went ahead, otherwise v2 +
guard. It is gated against SFT v2 by `configs/gate.yaml` unchanged (complete run; net +2 tasks;
Tool/Arg/Loop findings per conversation up no more than 10%; no domain losing 2 or more), plus a
**fifth check** (SEL-97): the frozen checker's made-up-intent findings per conversation may not
rise. This is the gate's third decision on dev (`max_cycles: 3`); there is no fourth.

**The Stage 2 winner** is the candidate if it passes. If nothing passes by Sun 22 Nov, 23:59 UK,
the winner is v2 + guard, reported as not having passed the gate.

## Test arms, declared (SEL-105, run once each)

100 tasks × 2 each, run Sun night, scored Mon. In this order if GPU time runs short:

1. **The Stage 2 winner.** If it is the retrained model, without the guard.
2. **c1**: `heal-trajectory-cycle-c1`, the adapter the automated loop promoted in Stage 1.
3. **P**: base Aya on our vLLM server (Arm A on the same server as Arm B).
4. **PC**: base Aya with the "careful" prompt on our server (Arm C on the same server as Arm B).
5. **The winner with the guard**, only if the winner is the retrained model and there is time.

**Crash rule:** conversations that crashed are re-run; finished conversations are never re-run.

## Retention (SEL-84, SEL-107)

Rule and benchmarks as fixed in `configs/retention.yaml` (committed 20 Nov, 13:56, before the first
result at 17:07): IFEval (541, strict prompt-level), GSM8K (1,319, 5-shot, strict match), and
Global-MMLU-Lite (400 each in en, fr, ar, hi, ja), full test sets, temperature 0, same server.
Healing "costs nothing" on a benchmark unless the paired 95% interval of the difference over the
same items lies entirely below zero. Drops and rises are both reported. Base vs v2: done
(`results/retention-base-vs-sft2.metrics.json`). Winner: run Sun night with the test arms.

## ROI reference (SEL-86)

GPT-4.1 as the agent (`configs/arms.gpt41.yaml`), test 100 × 2, a reference, not a candidate: it
trains nothing. Cost per conversation and per solved task against base and healed Aya on one H100.
GPT-4.1 plays both agent and customer; say so wherever it is reported. The agent runs through
OpenRouter; the customer is the pinned snapshot through OpenAI, as for every arm, with τ²-bench's
judges live.

The first run (launched 20 Nov, 17:19) is void and not reported as a result: its customer was
OpenRouter's unpinned GPT-4.1, and 147 of its 200 conversations crashed. It is re-run once with the
settings above; the cause of the crashes is recorded with the re-run.

## Classifier audit and the 37 (SEL-85, SEL-87)

Report only, no pass mark. Sheets from `scripts/labels/make_sheets.py` (seed 0), definitions in
`docs/experiments/LABELLING_GUIDE.md`. Two labellers (Charlie and Katie), separately, with the keys
closed until both have finished; agreement as Cohen's kappa. Precision: 60 findings (20 Tool, 20
Arg, 20 Loop, from 50 conversations); a finding is real if the claimed kind is correct as defined,
whatever the outcome. Recall: 40 conversations, flagged and not. The 37: one main error each
(`no_yes`, `nobody_asked`, `against_policy`, `reasoning`, `unclear`), sizing only, never used to
build the checker. Precision rests mostly on the base model's findings; false alarms could only
overstate v2's rate, so the reported drop is conservative.

## Launched before this file

- Retention, base vs v2 (SEL-84): rule fixed in `configs/retention.yaml` before the result.
- ROI reference, GPT-4.1 on test (SEL-86): launched 20 Nov, 17:19 with the wrong customer and
  mostly crashed; void, and re-run as declared under "ROI reference".

## Reported whatever it shows

Decision 1, the retrained model's validation score, Decision 2 and its reason, every declared test
arm, retention for three models, ROI, the audit and the 37, and meta-healing's candidates with
their scores, including any that failed.
