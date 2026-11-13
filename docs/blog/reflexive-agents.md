---
title: "Reflexive Agents: Teaching a 32B Model to Heal Its Own Hallucinations"
thumbnail: /blog/assets/reflexive-agents/thumbnail.png  # TODO: choose a thumbnail (e.g. the test-results chart)
authors:
- user: TODO-katie        # TODO: confirm names and Hugging Face handles
- user: TODO-charlie
- user: TODO-chiara
---

# Reflexive Agents: Teaching a 32B Model to Heal Its Own Hallucinations

Customer-service agents are being trusted with real write access: cancelling orders, changing
flights, editing accounts. When they hallucinate, it isn't a wrong sentence; it's a wrong action.
They call a tool they don't have, invent an order ID, or retry the same failing call until the
conversation runs out. Most systems treat these as terminal failures that need a human to step in.

We wanted to know whether an agent can **heal itself**: detect how it went wrong, get a stronger
model to correct the failing turn, retrain on the correction, and only keep the result if it
passes a gate. We built that loop for the hackathon on [τ²-bench](https://github.com/sierra-research/tau2-bench),
with **Aya Expanse 32B** as the student and **GLM-5.3-Flash** as the teacher, using the Lexsi Labs
stack ([AgentTune](https://github.com/Lexsi-Labs/AgentTune), AuditKIT and DECIDE).

**What we found:**

* **Healing works, and it's mostly about data.** Fine-tuning on the teacher's verified fixes took
  the student from **14.5% to 51.5%** of the 100 test tasks, and cut conversations with a
  hallucinated tool or argument from **95% to 10%**.
* **Asking nicely doesn't.** A "be careful not to hallucinate" instruction made no difference
  (13.0%, with 95.5% of conversations still hallucinating).
* **Reinforcement learning was harder than it looked.** Two GRPO runs on top of the fine-tuned
  model taught us more about reward design than they improved scores, and the second one
  showed something we didn't expect (below).
* **We're releasing** the verified teacher fixes, labelled by failure type, as a dataset.

### The setup: a student, a teacher, and a benchmark that checks the database

τ²-bench gives an agent a customer (simulated by GPT-4.1), a policy, and real tools over a
database in three domains: **retail**, **airline** and **telecom**. A conversation only passes if
the final database (and, in telecom, the customer's phone) ends up in exactly the right state.
That's what makes it a good test of hallucination: an invented argument doesn't just read badly,
it changes the wrong record.

We froze the splits on day two (**train 142 tasks, dev 36, test 100**) and gave each one a single
job: train tasks for training, a held-out slice of train for choosing between versions, dev for the
deployment gate, and test only for arms declared in advance.

**Choosing the teacher.** We screened 12 models on 30 train tasks, then ran the two finalists on
every train task. GLM-5.3-Flash passed **88% on the first try**, produced a verified fix for **158
of 178 tasks**, and cost **$11** in total. It tied Qwen3.8 27B on retail and airline and won on
telecom, the hardest domain, by **89% to 49%**.

![Teacher pass rate by domain](../experiments/img/phase2_by_domain.png)

*Aya Expanse 32B itself started as the teacher in our proposal, but unaided it solved 1 of 30
train tasks, so it became the student instead. Its 128K context also fits telecom's long
conversations.*

### What does a hallucinating agent look like?

Before training anything, we read the student's conversations. Aya passed **7 of 30** train tasks,
and the format wasn't the problem: there were **0 unreadable tool calls in 266 replies**. Every
failure was behaviour:

* calling tools it doesn't have (in telecom, the customer's own phone tools);
* inventing IDs, or passing placeholders like `"<order_id>"`;
* retrying calls that had already failed;
* batching calls that depended on each other.

To measure this at scale, we wrote a deterministic classifier that reads a conversation as the
agent saw it and labels every structural failure:

| Type | What it means | Example |
|---|---|---|
| **Tool** | a call to a tool the agent doesn't have | `check_apn_settings`, a phone tool only the customer can run |
| **Arg** | the right tool with invented, placeholder or ill-typed arguments | `find_user_id_by_email(email="something@example.com")` |
| **Loop** | repeating a failed call, the same call three times, or the same reply | a failing `modify_pending_order_items` sent 9 times |

On the test split, the untuned student (**Arm A**) had a Tool or Arg failure in **95% of
conversations**.

### Can you just tell it to be careful?

The cheapest fix is a prompt. **Arm C** is Arm A with a fixed instruction not to hallucinate. On
the 100 test tasks it scored **13.0%** against Arm A's **14.5%** (difference −1.5 points, 95% range
−7.0 to +4.0), and still hallucinated in **95.5%** of conversations. The model knows the rule; it
doesn't follow it.

### The loop: detect, fix, retrain, gate

```
 task ──► student (Aya Expanse 32B + LoRA) ──► conversation ──► DETECT & CLASSIFY (Tool / Arg / Loop)
              ▲                                                         │ failures
              │ promote if the gate passes                              ▼
        GATE (dev split) ◄──── RETRAIN (LoRA) ◄──── FIX: the teacher rewrites the failing turn
```

1. **Detect & classify.** The classifier finds the first failing turn and its type.
2. **Fix.** The teacher sees the conversation *exactly as the student saw it*, up to the failing
   turn, and writes that turn again. A fix is only accepted if it's a real turn, doesn't give up
   (no "I'll transfer you to a human"), and contains no Tool, Arg or Loop failure. On a first
   sample, the teacher fixed **6 of 6** failures on its first try, for about **$0.001** each.
3. **Retrain** the student's LoRA adapter on the fixes.
4. **Gate.** The new adapter is promoted only if it beats the current one on the dev split, under
   a rule agreed before any result.

Alongside the loop, a **runtime monitor** watches conversations as they run and flags
`no_progress`: three agent turns in a row that only repeat what it already said. On telecom it
fired on all 7 conversations that looped to the step limit and on none of the 7 that passed.

### Result 1: fine-tuning on verified fixes heals the hallucinations

Our first training method was the simplest one: supervised fine-tuning (QLoRA, rank 16) on the
teacher's verified fixes. Version 1 trained on 112 conversations and lifted dev success from **11%
to 38%**, but the gate rejected it on airline, where it now made changes customers hadn't asked
for. Version 2 added 34 verified airline conversations and reached **60% on dev**. On 30 train
tasks no version had seen, success went from **9.0** (base model) to **17.0** of 30.

We declared v2 as **Arm B** and ran all three arms on the test split, 2 samples per task:

| Test split, 100 tasks × 2 samples | Arm A (untuned) | Arm C (careful prompt) | **Arm B (healed)** |
|---|---:|---:|---:|
| Success (mean of 2 samples) | 14.5% | 13.0% | **51.5%** |
| 95% range | 9.5–20.0% | 7.5–19.0% | 43.5–59.5% |
| Conversations with a Tool or Arg failure | 95.0% | 95.5% | **10.0%** |
| Conversations with a loop (classifier) | 49.0% | 49.5% | **27.0%** |

**Arm B beats Arm A by 37.0 points** (95% range +28.0 to +45.5, paired on the 100 tasks). Every
success was re-checked by an independent scorer with no disagreements. The gain is largest in
retail (12.5% → 60%) and telecom (15% → 54%), and smallest in airline (17.5% → 30%), where the
remaining failures are about policy decisions rather than hallucinations.

TODO: a chart of this table for the post (and the thumbnail).

### Result 2: reinforcement learning, and what reward design taught us

The proposal was to heal with reinforcement learning (GRPO). We ran it twice, on top of the
fine-tuned v2, using single-turn GRPO: each training prompt is a conversation up to one agent turn,
the model samples that turn 8 times, and a reward scores each sample.

**The first run failed instructively.** We trained only on the teacher's *tool-call* turns. The
model learned that calling a tool was always rewarded, stopped replying to customers, and looped:
**0.5 of 30** on the validation tasks, against v2's 17.0. Its own training metrics looked healthy
throughout, because they only ever scored tool-call turns.

**The second run followed the loop properly.** We took 178 points where the base model failed
(Tool 47, Arg 103, Loop 63), had the teacher fix each one (**177 of 178 accepted**), and gave each
failure type its own reward: `tool`, `arg` and `loop`, plus agreement with the teacher's fix and a
guard against giving up. A teammate's review caught a way to game it (generic replies scored
almost as well as correct calls), so the failure rewards only count when the model picks the
teacher's kind of turn, and a drift guard watched the share of replies during training.

It did what we asked of it on single turns, and it didn't collapse. On whole conversations
(dev split):

| Dev, 36 tasks × 2 | Arm A | v2 (SFT) | v2 + GRPO |
|---|---:|---:|---:|
| Success | 6% | **60%** | 51% |
| Conversations with a loop | 61%* | 17% | **8%** |

*\*Arm A's dev loop rate includes its telecom rerun of 9 November; the other numbers are its original dev runs.*

GRPO **halved the loops** and cut repeated calls in telecom from **45% to 7%**, but lost to v2 on
success, and the gate rejected it. So we looked at the conversations it lost, and they weren't
hallucinations at all. They were **decisions**:

* in airline, it cancelled reservations nobody asked it to cancel (12 unrequested cancellations);
* in telecom, it walked customers through more steps on their phone, but skipped the fix on its
  own side of the account (turning on roaming, resuming a suspended line).

The rewards checked *how* a call was made: a real tool, real IDs, no repeats. A cancellation with a
real reservation ID passes all three, whether or not it should happen. **Hallucination rewards
taught the model to act confidently, not to act correctly.** A healing reward needs a decision check
too, for example only rewarding changes the task still expects.

### What we learned

* **Verified data beats instructions.** The fix that worked was showing the model hundreds of
  correct turns, each checked by the benchmark's own evaluator.
* **Decide the rule before you see the result.** Every version (SFT v1–v4, both GRPO runs) was
  judged by a rule written down before it ran. Several promising ideas didn't pass, and the rules
  kept us honest.
* **Give each split one job.** Choosing between versions on dev and reporting dev too would have
  overstated everything.
* **Watch what kind of turn the model takes,** not just whether its tool calls are valid. Both of
  our GRPO surprises were about *when* to act, not *how*.
* **Hallucination and decision errors are different problems.** Once the hallucinations were
  gone, the remaining failures were about policy: what to change, and when. That's the next frontier
  for self-healing agents.

### What's next

TODO: Charlie's additional full self-healing loop (in progress), and any further tests from
Charlie and Chiara.

The obvious next step for the loop is a reward that checks decisions as well as structure, and
scoring whole conversations rather than single turns.

### Try it

* **Dataset:** [`reagent-lexsi/tau2-golden-trajectories`](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories):
  verified teacher fixes for τ²-bench train tasks, labelled Tool / Arg / Loop. TODO: confirm it's
  public before publishing.
* **Code:** TODO: public repository link.
* **Adapters:** TODO: decide which adapters to publish (v2 is Arm B).

We'd love to hear what you try with it, especially other ways to reward the *decision* rather than
the call.

### Acknowledgements

TODO: thanks to Lexsi Labs for the hackathon and the stack, and the feedback for Lexsi Labs that
SEL-29 asks for (what worked in AgentTune, AuditKIT and DECIDE, and what we'd like to see).
