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

**Amended Sat 21 Nov, 11:38 UK, before it is measured: a hand-written "mismatch" rule (PR #76).**
Meta-healing's round 2 and its misfires pointed at changes that don't match the request (see the
result above); a person wrote the tight version. At 7 listed changes (retail: modify items,
exchange, return, cancel order; airline: update flights, update bags, cancel reservation) the
checker asks whether the change differs from what the customer asked for and agreed to, or from
the agent's own lookups. Checked by full GPT-4.1 (`gpt-4.1-2025-04-14`, temperature 0; amended
11:43, chosen before any result: mini read meta-healing's rules too loosely), through the same
machinery as meta-healing's rules (evidence must be in the conversation, errors never block), tool
results shown in full; live, with its own block message. "No yes" keeps gpt-4.1-mini. (Times corrected 11:50 to the commits' own;
earlier text gave 11:50 and 11:55 by mistake.) A run with gpt-4.1-mini, started from the 11:38
version, finished at 11:43, just after the switch was committed: it is reported beside the GPT-4.1
run and does not decide anything. The GPT-4.1 run is the measurement.

Amended Sat 21 Nov, before the GPT-4.1 run's numbers were seen: its first attempt stopped on one
checker error (a runaway answer, invalid JSON), after printing 5 flagged passes but no rates. As in
the guard, a checker error counts as not flagged; the count is reported and each rate is also given
at worst (every error flagged). The bar is read overall: at most 5 of the 55 passes, using the
at-worst count. Answers are capped at 4,000 tokens. The rule's wording is unchanged.

**Result, Sat 21 Nov 11:53 UK (GPT-4.1, no checker errors): it meets the bar and joins the
checker.** On SFT v2's 112 train-task conversations, "mismatch" flags 5 of 55 passes (9%, 95% CI
4–20%; all 5 retail, retail 5/21), 14 of 57 failures (25%) and 7 of the 24 failures no Tool / Arg /
Loop finding explains (29%). "No yes" alone: 3/55 passes, 4/57 failures. Either rule, as the guard
will run them: 6/55 passes (11%), 15/57 failures (26%). Injected errors: 37/40 caught (92%); the
same changes uncorrupted, 8/40 flagged (20%). Of the 5 flagged passes, read by hand, 1 is a real
mismatch the tool then rejected (retail 92, turn 29: an item from another order) and 4 are the
checker's reasoning slips (retail 109, 104, 11, 81). Expect some wrong blocks in retail in
Decision 1. The gpt-4.1-mini run, for comparison: passes 6/55, failures 21/57, injected 39/40 with
17/40 originals flagged. These are in-sample rates: the rule was written after reading v2's
train failures. Decision 1, on validation, is its first out-of-sample test. It joins the frozen checker only if it flags at most 1 in 10 of SFT v2's train-task
passes (`just rule-rates`), and of base Aya's (Arm A) train-task passes when those are run too
(validation tasks dropped). Failures flagged, unexplained failures flagged and an injected-error
test (corrupted changes from passing conversations: caught, and originals flagged) are reported
whatever they show. Measured once; the wording is not tuned on the result.

**Amended Sat 21 Nov, 22:01 UK, before any v2 code, data or result: meta-healing v2, and the
freeze moves to Sun 22 Nov 16:00 UK.** v1's result above stands. v1 had three diagnosed gaps: the
proposer never saw what the right actions were, tool results were cut to 600 / 1,200 characters,
and there was no feedback before the held-out test. Its held-out third (5 unexplained failures)
was also too small. v2, decided now:

- **Data.** Learning set: SFT v2's 112 train-pool conversations (already read by v1 and by the
  mismatch measurement). Held-out set: fresh conversations, run tonight: SFT v2 on the same 112
  train-pool tasks, 2 new samples each, guard and loop monitor in log mode
  (`S2-<domain>-fresh.jsonl`). Same tasks, new conversations: a rule that only works by knowing a
  task's details would pass here and not in Decision 1, so the review reads rules for generality.
  No validation, dev or test task is used.
- **Proposer.** GPT-4.1 (`gpt-4.1-2025-04-14`). On the learning set only, it sees each failure's
  expected actions (τ²-bench's) and full tool results. Its prompt names no error kind. The rules
  still read only the conversation, the call and the policy.
- **Feedback.** Up to 3 rounds on the learning set: propose, score every candidate there, show the
  proposer what each caught and wrongly flagged, refine. At most 8 candidates per round. Only the
  last round's candidates are scored on the held-out set, once.
- **Checker.** Full GPT-4.1 at temperature 0, tool results up to 6,000 characters, evidence must be
  in the conversation, an error counts as not flagged (with the worst case reported).
- **Unexplained, for proposing:** no Tool / Arg / Loop and no live "no yes" finding. "Mismatch" is
  left out, so we can see whether v2 rediscovers it: for each candidate we report the share of
  held-out failures "mismatch" flags that it also flags.
- **Accepted only if, on the held-out set:** it flags at least 3, and at least 10%, of the
  held-out failures that no frozen rule catches (Tool / Arg / Loop, "no yes" and "mismatch"); it
  flags at most 1 in 10 held-out passes, at worst; and a person reads 5 conversations it flags
  (the added catches first) and finds at least 4 real errors of the kind the rule names. At most 3
  accepted. If none passes, that is the result.
- **Also reported:** "mismatch" on the held-out set, its first out-of-sample rates.
- **Freeze: Sun 22 Nov, 16:00 UK, a hard deadline.** Accepted v2 rules join the checker if they are
  accepted by then; otherwise the freeze goes ahead with "no yes" and "mismatch", and v2 is
  reported separately. The freeze does not move again. Decision 1 runs after it.
- **The outer loop.** One full turn (checker rules → guard → guard fixes → retrain → gates) is in
  scope. Further turns are reported as next steps unless the timetable allows them; any extra turn
  is judged on fresh train conversations, and validation is used once, for Decision 1.

**Freeze: Sat 21 Nov, 23:59 UK; moved to Sun 22 Nov 16:00 UK, above (SEL-95).** The checker is tagged `stage2-checker-frozen`; the rule
list and each rule's live / after-the-fact mark are added below this line in the same commit.
Nothing changes the checker after the tag. If meta-healing isn't ready, the freeze goes ahead with
the "no yes" rule only, and meta-healing's rules are reported offline as a separate result.

**Frozen Sun 22 Nov, 12:36 UK, in the commit tagged `stage2-checker-frozen`.** The checker for every
Stage 2 guarded run (Decision 1, the guard-fix collection, the mismatch fixes, the dev gate's fifth
check, and any guarded test arm):

| Rule | Where | Checks | Model | Mark |
|---|---|---|---|---|
| Tool / Arg / Loop | `reflexive.hallucination` | every call | none (deterministic) | after the fact (the loop monitor stays in log mode) |
| `no_yes` | `reflexive.intent`, default `RULES` | retail and airline changes, telecom `refuel_data` | word match, then `gpt-4.1-mini-2025-04-14` | live |
| `mismatch` | `reflexive.mismatch`, `OPTIONAL_RULES` | 7 listed retail / airline changes | `gpt-4.1-2025-04-14` | live, own block message |

Runs opt into "mismatch" with `scripts/arms/rollout.py --rules mismatch` (PR #79); each record
notes `checker_rules`. No meta-healing rule is in the checker: v1 and v2 accepted none, and v2's
domain-scoped re-score (secondary) also passes none.

Rates measured with this checker (train-pool, 112 conversations; re-run after the review fixes):
"no yes" 4/57 failures, 3/55 passes (words only: 13/57, 10/55); "mismatch" 14/57, 5/55; either
15/57, 6/55; injected errors 29/40 caught with the original not flagged (37/40 flagged in all), 8/40
originals flagged. Held out (224 fresh): "no yes" 4/113, 2/111; "mismatch" 29/113, 7/111; with Tool /
Arg / Loop, 78/113 failures (69%) and 22/111 passes (20%). Meta-healing v2's domain-scoped re-score:
the closest, `retail_transfer_to_human_agent_on_policy_limit`, adds 4/14 retail failures and flags
6/43 retail passes (14%); none meets the bar.

**Re-frozen Sun 22 Nov, 13:04 UK, as `stage2-checker-frozen-2`, before 16:00 and before any
validation run.** One change to "no yes" (PR #80): a refusal after a polite or hesitant opening
("Please stop, I changed my mind.", "Sorry, no.") now withdraws an earlier yes, and "changed my mind"
counts anywhere; before, only a refusal at the very start of the message did. Found while checking
the review fixes. Its train rates were re-run with the same bar (at most 1 in 10 passes): unchanged ("no yes"
3/55 passes, 4/57 failures; words only 10/55, 13/57; "mismatch" 5/55), since no train conversation has a
withdrawal after a polite opening. Meta-healing v1's split is unchanged by the review fixes (checked at 12:56: the same
74 / 38 conversations and unexplained failures per domain). The first tag, `stage2-checker-frozen`,
stays where it is; every Stage 2 guarded run uses `stage2-checker-frozen-2`, which also has the
`S2G` arm (PR #81: v2 with the guard, its own key so its records never mix with v2's). A judge-only "no yes"
(the LLM decides every time, the word match only recorded) is noted as future work.

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
**No-go:** skip retraining; v2 + guard goes to Decision 2. (Amended Sat 21 Nov 23:28, see
Retraining: with an accepted meta-healing rule, a no-go retrains on meta-healing's fixes only.)

## Retraining (SEL-100, SEL-102, SEL-103)

- Fixes: v2 + guard (act mode) on the train tasks. Kept only if τ²-bench passes them, with
  `assert_not_test()` on every task.
- Too few: if fewer than 30 conversations pass, report the count, skip retraining, and v2 + guard
  goes to Decision 2. No top-ups from any other data.
- Recipe: from the base model, exactly SFT v2's recipe (`configs/train/sft_aya32b_v2.yaml`: 2
  epochs, learning rate 1e-4, same QLoRA settings), on v2's data plus the fixes as their own group.
  One change from v2: the added fixes. Adapter published with its `lexsi_provenance.json`.
- First score: validation 30 × 2 **without** the guard, reported whatever it shows.

**Amended Sun 22 Nov, 10:14 UK, after meta-healing v2's held-out score and before its result is
recorded: transient checker errors are re-asked once.** The score (Sun 00:35–01:10) had 51 checker
errors on each of the two telecom whole-conversation rules and 1 on the frozen rules; no rule met the
bar. A check at 10:14 re-asked 5 of the telecom errors and all 5 answered, so the errors were
transient (most likely rate limits on long telecom prompts, 8 calls at a time). As the crash rule does
for crashed conversations, `score` is run once more: every answer already given is replayed from the
cache unchanged, and only the calls that errored are asked again. Rules, data, bar and models are
unchanged. Both runs are reported; the second decides. No further re-runs.

**Meta-healing v2 result, Sun 22 Nov 10:20 UK: no rule accepted.** Second run (2 calls at a time):
every v2 rule answered on every conversation, with 1 checker error left on the frozen rules. Held
out: 224 fresh conversations, 113 failures, 111 passes; 35 failures no frozen rule catches.

- The closest, `retail_transfer_to_human_agent_on_policy_limit` (whole conversation), added 7/35
  (20%) but flagged 14/111 passes (13%, bar 10%); it also flags 17 of the 29 failures "mismatch"
  catches, so it fires on failure in general more than on one error. Every other rule added 0 or 1.
- Rediscovery: two rules are slices of "mismatch" (5/29 and 4/29); none is the general rule.
- First run (errors not re-asked): the same verdicts; the two telecom rules then had 51 errors each.
- "Mismatch" out of sample: 29/113 failures (26%), 7/111 passes (6%), as in sample (25%, 9%).
  "No yes": 4/113, 2/111. With Tool / Arg / Loop, the frozen checker spots 78/113 failures (69%,
  from 56% for Tool / Arg / Loop alone) and flags 22/111 passes (20%, from 14%).

Under the amendment of Sat 23:28, with no v2 rule accepted, retraining is as before it: only if
Decision 1 says go.

**Mismatch fixes, Sun 22 Nov 13:35 UK (SEL-117, `onpolicy_fixes.py --points mismatch`, #82): 39
fixes, so retraining runs whatever Decision 1 says.** Over v2's 112 train-pool and 224 fresh train
conversations, "mismatch" flagged a change in 43 failed and 12 passed ones (no checker errors); the
12 in passes get no fix, as pre-registered. GLM-5.3-Flash, told what was flagged in its own copy of
the prompt only, fixed 39 of the 43 (4 refused after 3 attempts), from 23 distinct tasks.

- **What the fixes are:** 27 ask or explain with no change (many correct the agent's own earlier
  mistake before acting: a wrong item's specs, a price per passenger, a misstated policy), 9 look
  something up first, 3 make the expected change. The answer key verified every change; a turn with
  no change passes that check, as in Stage 1. So the fixes teach the guard's caution, *check before
  acting*, more than the correct action. Only the fixed turn is verified, not the rest of the
  conversation.
- A check for fixes that say a change was made without making it found none among the 39. Neither
  `check_fix` nor any rule reads what a turn says, only its calls: "claimed but not done" is not in
  the taxonomy, and is a next rule to build.
- **What to watch:** the retrained model asking or stalling where it should act. The dev gate (no
  drop in success; the fifth check) covers it; average conversation length is reported too. The
  targeted measure, "mismatch" flags on the retrained model's dev conversations against v2's, is
  the test of whether the caution was learned.

**Amended Sun 22 Nov, 12:18 UK, after Chiara's review of PRs #74–#77 and before any re-run, merge
or freeze.** Her review fixed bugs in all four (25 issues, each with a test). Three affect results
already recorded, and are handled as follows, decided now:

- **"No yes" is the fixed version.** Five bugs: contractions ("isn't", "don't") and questions ("is
  that correct?") read as a yes; friendly openings ("Don't worry, go ahead") read as a refusal; a
  yes not tied to the change it answers; an empty judge answer blocking instead of counting as an
  error. The frozen checker uses the fixed rule. Its train rates (`just intent-rates`, with and
  without the judge) and the "no yes" and "either" rows of `just rule-rates` are re-run before the
  freeze, and replace the earlier numbers; both are reported.
- **"Mismatch" is opt-in, and opted in for Stage 2's guarded runs.** The rule is unchanged. It is
  no longer in the default `RULES`, so earlier published arms behave as before; Decision 1, the
  guard-fix collection and the mismatch fixes run with `checker_with("mismatch")`. It met its
  pre-registered bar, which is overall (5 of 55 train passes; 7 of 111 held-out passes). Retail
  alone was 5 of 21; Decision 1's check on wrongly blocked good changes tests exactly that, and is
  reported per domain. The injected-error count now credits a catch only when the original change
  is not flagged too (`caught`; the old count is kept as `caught_raw`).
- **Meta-healing v2's domain-scoped re-score is a secondary analysis and cannot change the
  verdict.** The review found that rules were scored on every domain, though each is named for one
  (e.g. `retail_transfer_to_human_agent_on_policy_limit`). The re-score that limits each rule to its
  own domain is reported beside this morning's result. The primary verdict stays as recorded: no
  rule accepted. If a rule crosses the bar only under the re-score, that is reported as a
  sensitivity result and a reason for v3, not an acceptance. Because the fixed "no yes" changes
  which held-out failures count as caught, the re-score also reports the frozen layers again.
- **Meta-healing v1 is not re-run.** It used the earlier "no yes"; that is noted beside its result.

**Amended Sun 22 Nov, 10:40 UK, before the freeze and before Decision 1 runs: retraining runs either
way, on fixes at the "mismatch" rule's findings.** The amendment of Sat 23:28 applies, with the
frozen "mismatch" rule in place of an accepted meta-healing v2 rule (it is the rule meta-healing v1
pointed to, tightened by a person, and the only new rule that held up out of sample). So:

- **Fix points:** every change "mismatch" flags in the train conversations (v2's 112 train-pool and
  224 fresh ones), failures and passes alike. GLM-5.3-Flash rewrites that turn, told the rule's
  description, up to 3 attempts, through Stage 1's checks; on these train tasks a fix is kept only
  if every change in it is one the task still expects. A wrong flag therefore yields no fix.
- **Retraining:** Decision 1 go: v2's data + guard fixes + mismatch fixes. No-go: v2's data +
  mismatch fixes. Fewer than 30 fixes in what would be used: report the count, skip retraining.
  The 30 is not lowered.
- **Targeted measure (reported, not a gate):** "mismatch" flags per conversation on the retrained
  model's dev conversations against v2's.
- **What it shows:** a loop that chooses what to fix without an answer key, as it would in
  deployment. On train tasks the answer key could find these mistakes anyway; this is a
  demonstration of the label-free loop, not a better source of data. An overall gain is credited to
  the loop as a whole.

**Amended Sat 21 Nov, 23:28 UK, before meta-healing v2 is scored and before Decision 1 runs:
meta-healing's fixes join retraining, and retraining no longer depends on Decision 1.**

- **Meta-healing fixes.** For each meta-healing v2 rule accepted by the freeze, every place it
  fires in the train conversations (v2's 112 train-pool conversations and the fresh ones) is a fix
  point: a call rule's flagged call; for a whole-conversation rule, the agent turn holding the
  rule's quoted evidence, else the agent's last turn. GLM-5.3-Flash rewrites that turn, told the
  rule's description, up to 3 attempts, through Stage 1's checks (`heal.fix_point`, `check_fix`,
  `to_sft_example`). On these train tasks a fix is kept only if every change in it is one the task
  still expects (`onpolicy_fixes`' rule). They are added as their own group, next to the guard's.
- **Retraining runs either way.** Decision 1 go: v2's data + guard fixes + meta-healing fixes.
  Decision 1 no-go: v2's data + meta-healing fixes. Too few: if the fixes that would be used number
  fewer than 30, report the count and skip retraining (v2 + guard to Decision 2, as above). If no
  meta-healing rule is accepted, everything is as before this amendment.
- **Same recipe** as above (from base, v2's recipe, the fixes as their own groups). Validation
  without the guard is still the retrained model's first score; Decision 2 is unchanged.
- **The targeted measure (reported, not a gate).** Each accepted rule's rate (conversations it
  flags) on the retrained model's dev conversations against v2's dev conversations: does the kind
  of error meta-healing found become rarer? An overall gain is credited to the loop as a whole, not
  to meta-healing alone: separating them needs a second retrain without its fixes, which is a next
  step.

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

## Stage 2b: training for an uplift over v2

**Amended Sun 22 Nov, 16:04 UK, after Decision 1 and before any Stage 2b run.**

**Decision 1's result (no-go).** v2 + guard (S2G), validation, 2 samples: 15.0 of 30 (retail 6.5,
airline 1.5, telecom 7.0) against v2's 17.0 (6.5, 2.5, 8.0). Condition 1 fails, so it is a no-go
whatever the others show. The guard blocked 5 replies in 2 of 60 conversations, both retail returns
(task 10 sample 1, which v2 passed and S2G failed; task 25 sample 0, failed by both); 2 more were
overridden. Wrongly blocked good changes 0 of 30 passes; no checker or judge errors; 0 crashed. By
Stage 2's rules no guard fixes are collected, and the 21 full-conversation mismatch fixes are under
30, so nothing is retrained. That is Stage 2's pre-registered result and stays the headline for it.

**Why Stage 2b.** The evidence across Stage 1 and 2 says v2 is under-trained, and that validation at
2 samples can't see a gain of +2:

- v2 passes 55 of the 112 train tasks it trained on (SFT.md, v4), each with a verified teacher
  conversation in its data.
- More steps on the same data helped (v1 14.0, SL 15.5); v3 and v4 added data at a fixed 74 steps
  and telecom fell in both, which SFT.md puts down to fewer passes over each telecom example.
- Held-out loss was still falling at the end of every SFT run; the recipe (2 epochs, rank 16,
  learning rate 1e-4) was never varied except SL's steps.
- v1's two validation samples differed by 4; v3 and v4 (17.0) can't be told apart from v2.
- Whole verified conversations train well where single turns don't (heal cycles: 6 → 22 against
  4 → 4 on dev), which is why the 21 mismatch fixes are whole conversations.
- 4 of v2's 26 validation failures ran past 32K context or into too many errors; the organisers
  suggested quantising to free memory for a longer context, and checking the metrics hold.

**The arms, fixed now: at most these five reach validation, in this order.** Each is trained from
the base model, scored on the 30 validation tasks × **4 samples** (8: amended below), served as v2 was unless it says.

| Arm | What changes from v2 | Config |
|---|---|---|
| S2E4 | 4 epochs (about 148 steps) | `configs/train/sft_aya32b_v2_e4.yaml`, `configs/sft/arms.s2e4.yaml` |
| S2R64 | 4 epochs and LoRA rank 64 (alpha 128) | `configs/train/sft_aya32b_v2_e4_r64.yaml`, `configs/sft/arms.s2r64.yaml` |
| S2Q | v2 itself, served with FP8 weights and 64K context | `configs/sft/arms.s2q.yaml`, `QUANTIZATION=fp8 MAX_LEN=65536` |
| S2D | the best recipe of S2E4, S2R64 and v2, on v2's data + v4's 29 on-policy fixes + the 21 mismatch fixes + extra verified teacher passes for retail and telecom (attempts at temperature 0.7, as v2's airline extras), minus any audited conversation confirmed bad | its own PR, before it trains |
| S2P | the practice round: the best model so far solves the train tasks outside validation 4 times each at temperature 0.7; its clean passes the checker doesn't flag, plus a teacher conversation for each task it never passes, are added to S2D's data; retrained from base with the same recipe | its own PR, before it trains |

**The baseline:** v2 (S2) gets samples 2 and 3 on validation with its own config, so every
comparison is 4 samples against 4. **The rule, for every arm but S2Q** (`scripts/arms/stage2b_compare.py
S2 <arm>`): it meets the rule only if its gain over v2 is at least +2.0 of 30, the paired bootstrap
95% range over the 30 tasks is above 0, and no domain falls by more than 1.5. The "best recipe" for
S2D is the arm of S2E4 and S2R64 with the larger gain that meets the rule, else v2's recipe.
**S2Q is a serving check, not a candidate:** it is adopted for S2D and S2P if its total is no more
than 1.0 below v2's and fewer of its conversations end on the context limit; otherwise they are
served as v2 was.

**The winner** is the arm that meets the rule with the largest gain (a tie goes to fewer changes).
It goes through the dev gate (Decision 2, as written) and one declared test run (100 × 2). If no arm
meets the rule, v2 stays, and every arm is reported with its numbers.

**Amended Sun 22 Nov, 16:32 UK, after v2's 4-sample baseline and before any candidate is
scored: 8 samples, not 4.** v2 (S2) on validation, 4 samples: 16, 18, 19 and 12 of 30 (mean 16.25;
retail 7, 6, 7, 5; airline 3, 2, 3, 2; telecom 6, 10, 9, 5). The samples' spread (SD 3.1) puts the 95%
range of a 4-against-4 difference at about ±4.3, so a real gain of +2 to +3 would usually fail the
rule's "range above 0". At 8 samples it is about ±3. So v2 gets samples 4 to 7, and every arm that
reaches validation is scored on 8 samples (`--samples 8`; `stage2b_compare.py` defaults to 8). The
rule is unchanged: +2.0, a 95% range above 0, no domain down more than 1.5. S2Q's serving check
uses the same 8 samples. Decision 1's S2G (15.0, 2 samples) sits inside v2's range of 12 to 19.

**Amended Sun 22 Nov, 17:01 UK, before any candidate is trained or scored: the self-practice loop
comes first; S2R64 is dropped; the arms are S2E4, S2P1, S2Q, S2D and S2P2.** Stage 2 healed only
what a detector found (about a quarter of failures; 113 failures → 21 verified fixes, too few to
retrain). The loop heals what τ²-bench's verifier finds, on every task, from the student's own
whole conversations: a kind of data no SFT version has had (v2 and v3 are the teacher's
conversations, v4 single teacher turns). Round 1 (S2P1, SEL-23), fixed now:

- **Practice** (`scripts/heal/practice.py run`, arm S2PR, `configs/sft/arms.s2pr.yaml`): v2 at
  temperature 0.7 on the 112 train tasks outside validation, 2 attempts each, guard off.
- **Verify** (`practice.py build`): kept only if τ²-bench passes it, it is clean
  (`golden.why_not_clean`), and the frozen "no yes" rule flags no change; at most 2 per task.
  "Mismatch" isn't used as a filter: the training-data audit (below) found at least 9 of its 12 flags
  on v2's 161 verified training conversations were its own errors. Counts and reasons are reported.
- **Heal:** a task the student never passes keeps the teacher's conversation already in v2's data.
- **Retrain** from base with v2's recipe exactly (`configs/train/sft_aya32b_p1.yaml`, 2 epochs), on
  v2's data plus the kept practice conversations (group `practice`), and score S2P1 on validation
  at 8 samples (`configs/sft/arms.s2p1.yaml`).
- **Read:** the rule against v2, as every arm. Also reported: S2P1 against S2E4, which has about as
  many training steps on v2's data alone, so the difference is the practice data's own effect.
- **Round 2 (S2P2),** only if it can finish before the test run must start (Mon 22:00 UK):
  practice with S2P1's model, the same filter, trained with the recipe (v2's or S2E4's) that met the
  rule with the larger gain, else v2's. The chart is validation success against round.
- **S2R64 is dropped**, to give the GPU to the loop. **S2Q** runs Monday on whichever pod is free.
  **S2D** runs only if there is GPU time before Mon 22:00. Neither changes the rule.

**Amended Sun 22 Nov, 21:11 UK, after S2E4's validation result and before any Stage 2 or 2b test
conversation: S2E4 goes to test as a declared arm.** S2E4 on validation, 8 samples: 18.50 against
v2's 16.00 (retail −0.88, airline +1.38, telecom +2.00), gain +2.50, 95% range −0.75 to +5.88.
It meets "gain at least +2.0" and "no domain down more than 1.5" and fails "95% range above 0", so it
**does not meet Stage 2b's rule**, and that verdict stands and is reported. Validation's 30 tasks
can't separate a gain of this size from luck; test's 100 tasks × 2 are the better check, and test
hasn't been opened in Stage 2. So, fixed now:

- **S2E4 runs once on test** (100 × 2, `scripts/pod/overnight.py`, its config, the guard off),
  compared with v2's Stage 1 test run (Arm B, 51.5%) paired on the same tasks, with a 95% bootstrap
  range over tasks. It is reported whatever it shows, labelled as an arm that did not meet the
  validation rule. It skips the dev gate: the gate is for a winner of the rule, and S2E4 isn't one.
- **S2P1, if it meets the rule on validation,** also runs once on test, the same way, then the dev
  gate as written. Both test results are reported against v2; neither is chosen after the fact. A
  claim of an uplift over v2 on test needs a 95% range above 0 for that arm.
- **S2Q moves to Monday** on whichever pod is free; its role is unchanged.
- `overnight.py` reruns crashed conversations until every task has both samples scored, then stops
  the pod; if a run is still incomplete after its retries, it stops the pod anyway and the run is
  resumed later. No test conversation is opened until every Stage 2 test number is in.

**Amended Mon 23 Nov, 23:29 UK, after S2E4's 2-sample test result and S2P1's validation result,
before any further test conversation: a declared test extension for S2E4 against v2.** Results so
far: S2E4 on test, 2 samples: 59.50% against v2's 51.50% (retail −1.50, airline +2.50, telecom +7.00
tasks), gain +8.00 points, 95% range −1.00 to +16.50, so no uplift is claimed from it. S2P1 on
validation: 17.9 against 16.0 (+1.9), short of the rule, so its test was skipped, as fixed. Test at
2 samples can only confirm gains of about +10 points; the uncertainty is mostly the customer
simulator's, which more samples reduce. So, fixed now:

- **v2 (S2) and S2E4 each get test samples 2 to 7** (`scripts/pod/extend_test.sh`: same configs,
  guard off, the same 100 tasks), for 8 each.
- **The claim** is judged on all 8 samples: an uplift over v2 on test is claimed only if the paired
  95% bootstrap range over the 100 tasks is above 0. Reported beside it, whatever they show: the
  2-sample result above, and samples 2 to 7 alone (v2's samples 0 and 1 are Stage 1's run; 2 to 7
  of both arms are run now with the same code), per domain.
- **Disclosed:** this extension was decided after seeing the 2-sample result (about 1 in 20 short of
  the bar). That is sequential testing, and it makes a false positive somewhat likelier than 5%;
  the paper says so. S2P1 is not extended: it didn't meet the validation rule and hasn't been on test.
- **Scope:** nothing else runs. About $60–100 of API calls; past the $200 total only with the team's agreement.

**The training-data audit (meta-healing on our own data).** `scripts/data/audit_training_data.py`
runs the frozen checker ("no yes" and "mismatch") over every state-changing call in v2's training
conversations. Its counts are reported whatever they show. Flagged conversations are read by two of
us, separately, before S2D's data is built and before S2E4's or S2R64's validation results are read;
only those both call a real unconfirmed or mismatched change are removed from S2D's data.

**Amended Sun 22 Nov, 17:05 UK, before it is run on anything: the hybrid judge (the organisers'
"LLM-as-a-judge with rule-based exceptions").** Stage 1's Tool / Arg / Loop classifier is all rules;
Stage 2's checker asks an LLM about one call at a time. The audit (below) showed how that judge
fails on verified-good conversations: it judges one call of a multi-call change as the whole (6 of
12 flags), its verdict contradicts its reasoning (2), it misreads prices (1). The hybrid judge
(`scripts/labels/hybrid_judge.py`, `gpt-4.1-2025-04-14`, temperature 0, cached) reads the policy and
the whole conversation once, after it ends, never the answer key, with facts computed by code for
exactly those errors: every change made, in order, with the customer's last message before it and
whether the tool accepted it; every variant's price and options from the lookups (retail); the Stage
1 classifier's findings. It answers reasoning first, then the verdict, turn, kind and a quote; an
error counts only if the quote is in the conversation and the turn is an agent turn.

- **Scored against the answer key,** for the first time for any of our detectors: τ²-bench's verdict,
  and for a failure the turn where the agent's changes first leave the expected ones (the fix step's
  `fix_point`). Against three baselines: the rules, the frozen checker, and both together (what the
  guard and the gate used).
- **Development run:** v2's train-pool conversations (`--split trainpool`, 112), for plumbing only:
  the prompt and the facts are fixed by this commit and don't change after it. If more than 1 in 20
  answers are invalid or errors, the plumbing is fixed and the development run repeated, once.
- **Held-out run, once:** the 224 fresh conversations (`--split fresh`), validation tasks left out.
- **The bar:** the hybrid judge beats the stack if, on the held-out run, it flags at least as many
  failures as rules + checker together and at most 1 in 10 passes. Also reported, whatever they
  show: per domain, at the true turn, within 4 messages, the error kinds, and every invalid answer.
- **If it meets the bar,** it may replace "no yes" as round 2's practice filter (S2P2), with
  "flagged" as the filter; that choice is made from the held-out numbers before round 2 practises.

**Also reported, not gates:** S2E4 on the train tasks outside validation (1 sample) against v2's
55 of 112, the direct test of under-training; the checker's flags per conversation on each arm's
validation conversations against v2's (the targeted measure); training and held-out loss per arm.

**Disclosed:** the validation tasks have now judged Decision 1 and judge up to five Stage 2b arms,
so a winner's validation gain is optimistic. Dev (Decision 2) and test are the held-out checks.
Budget as above ($200 in total for Stage 2 and 2b).

### Stage 2b's results

**Amended Tue 24 Nov, 09:13 UK: the results, as they came in, and the consent check, fixed before it
runs.** Every number below is from the runs' own files (`results/`), read with the scripts named.

| What | Result | Read with |
|---|---|---|
| v2 on validation, 8 samples | **16.0** of 30 (samples 16, 18, 19, 12, 14, 17, 16, 16) | `samples.py` |
| Decision 1 (S2G, 2 samples) | 15.0: no-go (above) | `samples.py`, `guard_report.py` |
| S2E4, validation, 8 samples | 18.50 vs 16.00, **+2.50** (95% −0.75 to +5.88): does not meet the rule | `stage2b_compare.py` |
| S2E4, test, 2 samples | 59.50% vs 51.50%, +8.00 (95% −1.00 to +16.50): no uplift claimed | `stage2b_compare.py --split test --samples 2` |
| **S2E4, test, 8 samples** | **60.00% vs 50.12%, +9.88 points (95% +3.88 to +16.00): uplift claimed**, by the 23:29 amendment | `stage2b_compare.py --split test --samples 8` |
| S2E4, test, samples 2 to 7 only | 60.17% vs 49.67%, +10.50 (95% +4.00 to +17.00) | `--samples 6 --min-sample 2` |
| S2E4 by domain, test, 8 samples | retail 22.38 vs 22.38 (0.00); airline 7.62 vs 7.00 (+0.62); telecom 30.00 vs 20.75 (**+9.25**) of 40/20/40 | same |
| Round 1 practice (S2PR) | 224 conversations, 108 passed; 100 kept (2 flagged by "no yes", 6 not clean); 74 of 112 train tasks passed at least once | `practice.py build` |
| S2P1, validation, 8 samples | 17.9 vs 16.0, **+1.9**: short of +2.0, so not on test (skipped by `overnight.py --if-meets`) | `samples.py`, `stage2b_compare.py` |
| Training-data audit (v2's 161 conversations) | 16 flagged: 12 "mismatch" (at least 9 the checker's own errors: 6 judged one call of a multi-call change as the whole, 2 verdicts against their own reasoning, 1 price misread; 3 undetermined), 4 "no yes" (telecom refuels the customer had asked for, with no explicit yes after the agent's confirmation). None removed | `audit_training_data.py` |
| Hybrid judge, held out (224 fresh) | failures flagged 42% (48/113), passes flagged 9% (10/111), at the true turn 11 of 44; rules 56% / 14% / 0 of 52; checker 27% / 8% / 15 of 28; rules + checker 69% / 20% / 15 of 67. **Bar not met**; 18 answers invalid (no exact quote); the one plumbing fix changed nothing | `hybrid_judge.py --split fresh` |
| S2E4 training | 148 steps, 56 min; held-out loss 0.854 → 0.351 (lowest 0.341 near epoch 2; v2 ended at 0.368) | `run_summary.json` |

**Not run, and why:** S2R64 (dropped at 17:01, for the loop); S2Q, S2D and round 2 (S2P2) (the
Mon 22:00 cut-off passed); the dev gate for S2E4 (it isn't a winner of the validation rule; 21:11);
the declared Stage 2 test arms (the "winner" v2 + guard, c1, P and PC: Decision 1 showed no
measurable effect of the guard, S2E4 is confirmed on test, and the time went to the write-up; Arm A
and Arm C on test from Stage 1 stand for the base model); retention on S2E4. All are reported as not
run.

**What the uplift is, and isn't.** It comes from training v2's own data for twice as long, nearly all
of it in telecom; retail and airline didn't move at 8 samples (the 2-sample airline gain was mostly
noise). S2E4 went to test as a declared arm that didn't meet the validation rule, and the extension
to 8 samples was decided after the borderline 2-sample result: both are disclosed, and make the
nominal 5% error rate somewhat optimistic. One model, one benchmark, 100 test tasks.

**The consent check, fixed now, before it runs** (the dev gate's fifth check, SEL-97, applied to
test, where S2E4 is the one confirmed candidate): `scripts/arms/consent_check.py S2 S2E4 --split
test` runs the frozen "no yes" rule over every scored test conversation of both arms (8 samples).
It **passes** if S2E4's "no yes" findings per conversation, averaged per task then over the 100
tasks, don't rise against v2's (paired change ≤ 0); the paired 95% range, per-domain counts and
findings per change are reported beside it. "Mismatch" isn't run (the audit; cost). Run once.

## Stage 2c: after the extension to Sat 28 Nov

**Amended Tue 24 Nov, 17:50 UK, before any of these runs.** The deadline moved to Sat 28 Nov, AoE, so
Stage 2b's closed scope reopens for six runs, fixed now. Nothing else runs. No result below changes
any Stage 2b result above; each is reported whatever it shows.

1. **SEL-121, does self-practice add to training longer?** S2P1E4: round 1's data
   (`data/train/sft-p1.jsonl`: v2's 146 + 100 verified practice conversations) with S2E4's recipe,
   4 epochs (`configs/train/sft_aya32b_p1_e4.yaml`, `configs/sft/arms.s2p1e4.yaml`). Scored on
   validation and on test, 8 samples each; **the test run happens whatever validation shows.** The
   **primary comparison is S2P1E4 against S2E4 on test**, paired over the 100 tasks, using S2E4's
   8-sample test run: self-practice is claimed to add to training longer only if the 95% range is
   above 0. Reported beside it: against v2 on test and on validation, per domain.
2. **The quantisation and context check (SEL-120, the organisers' point).** S2E4Q: S2E4 served with
   FP8 weights and a 64K context (`configs/sft/arms.s2e4q.yaml`), validation, 8 samples. **The
   metrics hold** if its mean is no more than 1.0 below S2E4's 18.50 (16-bit, 32K). Also reported:
   conversations ending on the context limit, GPU memory in use, and latency.
3. **Arm P on test (the organisers' "Cohere's model on GPU").** Base Aya on our vLLM server
   (`configs/sft/arms.base.yaml`, arm P: the local server, as P's validation runs used; amended Tue 21:55 from `arms.pod.yaml`, which reaches the pod through its public endpoint with a key and failed on every conversation before any was scored), test, 100 × 2, against Arm A (the same model through Cohere's API,
   Stage 1's test run), paired. Descriptive: no claim either way rests on it.
4. **Retention on S2E4** (`configs/retention.yaml`, now with `sft2e4`), with the rule as fixed for
   SEL-84: training longer "costs nothing" on a benchmark unless the paired 95% interval of the
   difference from base, over the same items, lies entirely below zero.
5. **S2E4 on the train tasks** outside validation, 1 sample at temperature 0, against v2's 55 of
   112: the direct test of the under-training diagnosis. Descriptive.
6. **What kinds of failure training longer fixed.** The hybrid judge, as frozen, on v2's and S2E4's
   failed validation conversations (8 samples each), with the answer key's kinds. Descriptive: the
   kinds per arm are reported, and no claim rests on them.

Plus the consent check, fixed at 09:13 and not yet run. Pods: 2, 3, 4, 5, 6 and the consent check on
pod A (`scripts/pod/night_a.sh`), 1 on pod B (`scripts/pod/night_b.sh`); each stops its pod.
Budget: about $110 to $140 of API calls and $45 of GPU, past the $200 total with the team's agreement.

### Stage 2c's results

**Amended Wed 25 Nov, 08:57 UK: the results of Stage 2c's runs, all complete** (pod A's queue ended
23:18, pod B's 03:16; each stopped its pod). Read with the scripts named; every run is reported.

| Run | Result | Verdict by the rule fixed at 17:50 |
|---|---|---|
| **1. S2P1E4 against S2E4, test, 8 samples** (primary) | 58.88% vs 60.00%, **−1.12** (95% −6.62 to +4.38); retail +1.38, airline +0.38, telecom −2.88 | **Self-practice adds nothing measurable to training longer** (range not above 0) |
| S2P1E4 against v2, test, 8 samples | 58.88% vs 50.12%, **+8.75** (95% +3.62 to +13.88); retail +1.38, airline +1.00, telecom +6.38 | Reported beside it |
| S2P1E4 against v2, validation, 8 samples | 17.88 vs 16.00, +1.88 (95% −0.62 to +4.75) | Reported beside it |
| **2. S2E4Q (FP8, 64K) against S2E4 (16-bit, 32K), validation, 8 samples** | 18.50 vs 18.50, **+0.00** (95% −2.25 to +2.25); retail +1.12, airline −0.12, telecom −1.00. Model weights in GPU memory: **32.23 GiB** (FP8) vs **60.35 GiB** (16-bit), from vLLM's startup logs | **The metrics hold** (not more than 1.0 below) |
| **3. Arm P (base Aya, our server) against Arm A (Cohere's API), test, 2 samples** | 11.50% vs 14.50%, −3.00 (95% −8.00 to +1.50); retail 0.00, airline +0.50, telecom −3.50 | Descriptive: the same within noise |
| 4. Retention on S2E4 | not run yet (SEL-107) | — |
| **5. S2E4 on the train tasks**, 1 sample | **60** of 112 vs v2's 55 (retail 19 vs 21, airline 11 vs 8, telecom 30 vs 26) | Descriptive |
| **6. Failure kinds, validation, 8 samples** | answer key: v2 112 failures (65 an unexpected change, 35 a change never made, 12 other) vs S2E4 92 (52, 32, 8); the hybrid judge's flagged kinds, v2 vs S2E4: wrong arguments 13 vs 6, stopped short 9 vs 3, wrong information 6 vs 8, against policy 4 vs 5, missed action 4 vs 0, unconfirmed change 2 vs 2, unrequested change 1 vs 0 | Descriptive |
| **Consent check** (fixed 09:13, Tue) | S2E4 0.010 "no yes" findings per conversation (6 of 800 conversations) vs v2's 0.152; **−0.142** (95% −0.310 to −0.035) | **Does not rise**: it falls |

**Read together:** the uplift over v2 comes from training longer, and holds with either data (S2E4
+9.88, S2P1E4 +8.75 on test, both ranges above 0); adding the model's own verified practice to the
teacher's data changes nothing measurable. The gain's shape is the same on train, validation and test
tasks (telecom up, retail flat), it comes mostly from fewer wrong changes, and the better model makes
fewer changes without the customer's yes, not more. Serving it in FP8 halves the weights' memory and
doubles the context without changing its score. Arm P ran with `configs/sft/arms.base.yaml` (amended
Tue 21:55); its telecom drop against the API's run is within noise and not investigated further.

### Stage 2d: a second training seed

**Amended Wed 25 Nov, 11:48 UK, before the run: S2E4 trained again with a different seed, and the
rule that says whether its gain replicates.** S2E4's +9.88 on test (95% +3.88 to +16.00) rests on
one training run, and training is random: the seed sets LoRA's starting weights and the order of
the examples. So the same recipe is trained once more, with one change, the seed (0 → 1):
`configs/train/sft_aya32b_v2_e4_s1.yaml`, arm **S2E4S1** (`configs/sft/arms.s2e4s1.yaml`), same
data, epochs, settings, server, prompt, decoding and customer as S2E4.

- **Declared for test, run once:** 100 tasks × 8 samples (`scripts/pod/night_seed.sh`: train,
  serve, test, compare, stop the pod). No validation run first: the question is replication of
  a test result, not a new candidate.
- **The rule (primary): the gain replicates if S2E4S1 beats v2 on test with the paired 95% range
  above 0,** the rule S2E4's own test claim met (`stage2b_compare.py S2 S2E4S1 --split test
  --samples 8`).
- **Reported whatever it shows:** S2E4S1's success, its gain over v2 with the 95% range, by domain,
  and its difference from S2E4 (seed 0) with the 95% range.
- **If it does not replicate,** both runs are reported, and the headline becomes the average gain
  over the two training runs with the spread between them. Neither run is dropped.
- **Crash rule** as for every arm: conversations that crashed are re-run; finished ones never are.

## Launched before this file

- Retention, base vs v2 (SEL-84): rule fixed in `configs/retention.yaml` before the result.
- ROI reference, GPT-4.1 on test (SEL-86): launched 20 Nov, 17:19 with the wrong customer and
  mostly crashed; void, and re-run as declared under "ROI reference".

## Reported whatever it shows

Decision 1, the retrained model's validation score, Decision 2 and its reason, every declared test
arm, retention for three models, ROI, the audit and the 37, and meta-healing's candidates with
their scores, including any that failed.
