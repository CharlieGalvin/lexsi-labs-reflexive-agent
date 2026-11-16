# Future work: healing decisions, and more than one teacher

The hackathon result is that healing hallucinations works: on τ²-bench's test split, conversations
with a made-up tool or argument fell from 95% to 10%, and tasks solved rose from 14.5% to 51.5%.
This page plans the next step: the failures that are left.

## Where the remaining failures are

In 37 of Arm B's 55 failed dev and validation conversations, the Tool / Arg / Loop classifier finds
no hallucination at all. The agent used real tools and real values, but did the wrong thing with
them: it cancelled a reservation nobody asked to cancel, went against airline policy, or walked a
customer through phone settings and skipped the fix on its own side of the account.

Some of these are hallucinations in a wider sense. The research literature separates
*factuality* (inventing facts) from *faithfulness* (contradicting the instructions or context the
model was given). Our classifier measures the first, in the conversation: a tool or a value that
nobody gave the agent. The second covers acting on an intent the customer never expressed, or
against a policy the agent was given.

### What can be caught from the conversation and the policy

No answer key is needed for these:

* **Changes nobody asked for.** Every change the agent makes (cancel, refund, exchange, modify)
  should trace back to something the customer actually requested.
* **Acting without confirmation.** τ²-bench's policy requires the customer to say yes before any
  change to their account. Whether they did is a simple rule over the transcript.
* **Breaking a written policy rule.** For example, refunding a fare the policy says can't be
  refunded. An LLM judge given the policy document can check this.

### What still needs the answer key

Errors of judgement where the conversation is ambiguous, steps the agent skipped, and simply
stopping too early. These are reasoning mistakes, not made-up information, and a detector can't see
them without knowing what the task required.

## Plan: one week

1. **Day 1: sort the 37 failures by hand.** Count how many are unrequested changes, missing
   confirmations, policy breaks, or plain reasoning mistakes. This sizes the opportunity before
   anything is built. If most turn out to be reasoning, stop here.
2. **Days 2–3: build the detector.** A rule for "changed without confirmation", plus an LLM judge
   for "unrequested" and "against policy". Check it against τ²-bench's verdicts the same way we
   checked AuditKIT: does it flag the failures and stay quiet on the passes?
3. **Days 4–5: use it as a guard, not a reward.** When the agent is about to make a change the
   detector flags, block it and have the agent ask the customer first. Our third GRPO run
   (Result 3 in [the blog post](blog/reflexive-agents.md)) showed that rewarding a decision score teaches the
   model to game it: it drifted towards calling tools. A guard stops the bad action before it
   happens, and it targets exactly the failures that cost tasks. It extends the runtime monitor's
   `Guard` (`src/reflexive/monitor.py`), which today acts only on loops.
4. **Evaluate honestly.** This project's dev gate budget (`configs/gate.yaml`, 3 cycles) and its
   test split are both used up, so a fair claim needs fresh tasks: new held-out τ²-bench tasks, or
   a second benchmark. Write the pass/fail rule down before the run, as we did here.

**Why a guard rather than more training data.** SFT v2 already trained on the teacher's correct
decisions for these tasks and still gets them wrong, so more of the same data probably won't fix
them. (SFT v3 and v4 added data and didn't beat v2.)

## More than one teacher

Today every fix comes from one teacher, GLM-5.3-Flash. It passed 88% of τ²-bench's train tasks on
the first try and produced verified fixes for 158 of 178, so the loop can't teach the student the
20 it can't solve, and it only ever shows one way of solving each task.

* **Coverage.** In the teacher bake-off ([`TEACHER_PHASE2_RESULTS.md`](experiments/TEACHER_PHASE2_RESULTS.md)),
  Qwen3.8 27B (thinking off) passed 4 of the 138 shared tasks that GLM failed. A pool of teachers,
  each fix still verified by τ²-bench's own evaluator, can cover tasks no single teacher solves.
* **Variety.** Two verified solutions to the same task show the student more than one correct path,
  which may help it generalise rather than memorise one script.
* **Agreement as a signal.** Where several teachers agree on a decision, it's more likely right;
  where they disagree, the task is a good candidate for the decision detector above.

To try: run a second and third teacher on the tasks GLM failed and on the student's failures; keep
only verified fixes; train with and without the extra teachers on the same budget, and compare on
held-out tasks. Watch for mixed styles (one teacher's tool-call habits contradicting another's) and
cost: a thinking model writes many more tokens per turn.

## Check general ability before and after

Healing trains the student on customer-service conversations only. We didn't check whether that
changed what it can do elsewhere. Next time, run a few benchmarks away from the task on the base
model and on the healed adapter, with the same settings, through AuditKIT:

* **Instruction following:** IFEval.
* **General knowledge:** MMLU (or Global-MMLU for other languages, since Aya is multilingual).
* **Maths and reasoning:** GSM8K.
* **Another tool-use benchmark**, such as BFCL, to see whether less hallucination carries over.

Report both scores for each, with intervals. Two outcomes matter: a drop is a cost to state
alongside the gain (training on one domain often costs some maths), and a rise would be a side
benefit worth testing further. An agent that stops inventing values and repeating failed calls
might do better on other tool-use and reasoning tasks too. Plan for a few GPU hours, and decide the
benchmarks and the rule before running them.

## Questions to answer first

* How many of the 37 are each kind of failure? (Day 1 decides the rest of the week.)
* How often does the detector flag a pass? A guard that blocks correct changes costs tasks.
* Does a second teacher add tasks, or only more copies of tasks already covered?
* Did healing change general ability, up or down?
