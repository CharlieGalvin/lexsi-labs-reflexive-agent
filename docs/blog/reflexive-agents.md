---
title: "Reflexive Agents: Teaching a 32B Model to Heal Its Own Hallucinations"
thumbnail: /blog/assets/reflexive-agents/test-results.png
authors:
- user: katiemarieek
- user: cp89
- user: cg3849
---

# Reflexive Agents: Teaching a 32B Model to Heal Its Own Hallucinations

Agents are increasingly trusted with write access: cancelling orders, changing flights, editing
accounts. When they hallucinate, the result isn't a wrong sentence but a wrong action. They call a
tool they don't have, invent an order ID, or retry a failing call until the conversation runs out.

We wanted to know whether an agent can **heal itself**: detect how it went wrong, have a stronger
model correct the failing turn, retrain on the correction, and keep the result only if it passes a
gate. We built this loop on [τ²-bench](https://github.com/sierra-research/tau2-bench), with
**Aya Expanse 32B** as the student and **GLM-5.3-Flash** as the teacher, using the Lexsi Labs stack
([AgentTune](https://github.com/Lexsi-Labs/AgentTune), AuditKIT and DECIDE).

**What we found:**

* **Healing works, and it's mostly about data.** Fine-tuning on the teacher's verified fixes took
  the student from **14.5% to 51.5%** of test tasks, and cut conversations with a hallucinated tool
  or argument from **95% to 10%**.
* **Asking nicely doesn't.** A "be careful not to hallucinate" instruction changed nothing.
* **Reinforcement learning didn't beat fine-tuning.** But it showed what a healing reward has to
  measure: not just *how* the agent acts, but *whether* it should.

### The idea, without the jargon

Picture a new trainee on a customer help desk. They're quick and polite, but under pressure they
bluff: they quote an order number that doesn't exist, use a system they don't have access to, or
press the same broken button again and again. That's an agent **hallucinating**.

Now give the trainee a senior colleague who replays each call that went wrong, stops at the exact
moment it went off track, and writes down what should have been said. The trainee practises on
those corrections: that's **teacher–student training**. Before they go back on the phones, a
manager runs them through a set of test calls and signs them off only if they solve more than
before.

Spot the mistake, get the correction, practise, pass the check, and repeat: that's
**self-healing**. In our project the trainee is a 32B open model, the senior colleague is a
stronger model, and the manager is a benchmark with a rulebook fixed in advance.

### The loop: detect, fix, retrain, gate

![The self-healing loop, animated. The student agent, Aya Expanse 32B with a LoRA adapter, fails a conversation. 1, Detect: a classifier finds the failing turn and flags an invented argument, email="something@example.com". 2, Fix: the teacher rewrites the turn as "Could you confirm your email?". 3, Retrain: the student trains on 146 verified conversations. 4, Gate: dev success rises from 11% to 60%, so the new adapter is promoted and the student moves from adapter v1 to v2.](img/heal-loop.gif)

1. **Detect.** A classifier finds the first failing turn and its type: Tool, Arg or Loop.
2. **Fix.** The teacher sees the conversation exactly as the student did, up to that turn, and
   rewrites it. A fix is kept only if it's a real turn, doesn't give up, and has no Tool, Arg or
   Loop failure of its own. Fixes cost about $0.001 each.
3. **Retrain.** The student's LoRA adapter is trained on the fixes.
4. **Gate.** The new adapter replaces the old one only if it does better on the dev split, under a
   rule agreed in advance.

A runtime monitor also watches live conversations for loops. On telecom it flagged all 7 that
looped to the step limit, and none of those that passed.

### The setup

τ²-bench gives an agent a simulated customer, a policy and real tools over a database, in three
domains: retail, airline and telecom. A conversation passes only if the database (and in telecom,
the customer's phone) ends in exactly the right state. That makes it a good test of
hallucination: an invented argument doesn't just read badly, it changes the wrong record.

We froze the splits early (train 142 tasks, dev 36, test 100) and gave each one a single job.
Train is for training, a held-out slice of train for choosing between versions, dev for the
deployment gate, and test only for models declared in advance.

To pick the teacher, we screened 12 models, then ran the two finalists on all 178 tasks of
τ²-bench's own train split (our train and dev splits together). GLM-5.3-Flash passed 88% on the
first try and produced verified fixes for 158 of 178 tasks, for **$11** in total.

### What a hallucinating agent looks like

Untuned, the student passed 7 of 30 train tasks. Its tool-call format was fine (0 unreadable calls
in 266 replies), so every failure was behaviour. It called tools it didn't have, invented IDs,
retried calls that had already failed, and batched calls that depended on each other.

To measure this at scale, we wrote a classifier that labels every structural failure in a
conversation:

![Three failure types. Tool: the agent calls a tool it doesn't have, such as check_apn_settings(), a phone tool only the customer can run. Arg: it invents an argument, a placeholder, made-up or ill-typed value such as email="something@example.com". Loop: it repeats itself, sending the same failed call, the same call or the same reply, such as one failing call sent 9 times.](img/failure-types.png)

On the test split, the untuned student had a Tool or Arg failure in **95% of conversations**.

### Can you just tell it to be careful?

The cheapest fix is a prompt. With a fixed instruction not to hallucinate, the student scored
13.0% against 14.5% without it (a difference of −1.5 points, 95% range −7.0 to +4.0), and still
hallucinated in 95.5% of conversations. The model knows the rule; it doesn't follow it.

### Result 1: fine-tuning on verified fixes

Our first pass through the loop was by hand: we fine-tuned the student (QLoRA, rank 16) on the
teacher's whole verified conversations for the train tasks it failed. A first version raised dev
success from 11% to 38%. The gate rejected it on airline, where it now made
changes nobody had asked for. Adding 34 airline conversations fixed that, and reached **60% on
dev**. We declared this version as **Arm B** and ran it on the 100 test tasks, against the
untuned student (**Arm A**) and the careful prompt (**Arm C**):

![Test results by domain. Task success for arms A, C and B: all 100 tasks 14.5%, 13% and 51.5%; retail 12.5%, 10% and 60%; airline 17.5%, 20% and 30%; telecom 15%, 12.5% and 53.8%. Conversations with a Tool or Arg hallucination: 95%, 95.5% and 10%.](img/test-results.png)

![Test split, 100 tasks × 2 samples. Success: Arm A 14.5%, Arm C 13.0%, Arm B 51.5%; Conversations with a Tool or Arg failure: Arm A 95.0%, Arm C 95.5%, Arm B 10.0%; Conversations with a loop: Arm A 49.0%, Arm C 49.5%, Arm B 27.0%](img/test-table.png)

**Arm B beats Arm A by 37 points** (95% range +28 to +46). The gain is largest in retail and
telecom, and smallest in airline, where the remaining failures are policy decisions rather than
hallucinations.

### Result 2: the loop, run automatically

We then ran the whole loop on the GPU with no one in it: the student runs train tasks, the
classifier finds its failures, the teacher fixes them, an adapter is trained, and the gate decides
whether it's swapped in.

**First, at the grain we'd designed: fix only the turn that went wrong.** The teacher fixed
**24 of 24** failures, and the healed adapter made **71% fewer** Tool, Arg and Loop errors per
conversation. But it solved the same number of dev tasks, 4 against 4, so the gate rejected it.
It had trained on 16 rewritten turns, about 1% of the conversations' tokens: it learned not to
bluff at the start of a call, not how to finish one.

**Then, at the grain that worked by hand: the teacher's whole verified conversation for each
failure.** The base model failed 106 of 112 train tasks; the loop trained on the 136 verified
conversations for them, and the gate ran both adapters on the 36 dev tasks. The healed adapter solved
**22 against 6**, and made **87% fewer** Tool, Arg and Loop errors per conversation (8.2 → 1.0). The gate
promoted it and swapped it in, with no one in the loop.

| The loop, run automatically | Fix | Dev tasks (base → healed) | Hallucination findings per conversation | Gate |
|---|---|---:|---:|---|
| Cycle 1 | one rewritten turn per failure | 4 → 4 | −71% | reject |
| Cycle 2 | whole verified conversations | **6 → 22** | **−87%** | **promote** |

This second cycle feeds the loop what Result 1 taught, so a promotion was the expected outcome, and
22 of 36 matches what we got by hand (60% on dev). It shows the automated pipeline can do what we did
by hand, not new evidence that healing works: that evidence is the test result above. What the two
cycles show together is that **the grain of the fix decides whether healing moves success**: fixing
where an agent first goes wrong isn't enough; it has to learn how to finish the job.

### Result 3: what reinforcement learning taught us

We then tried to heal further with GRPO, starting from Arm B. Each training prompt is a
conversation up to one agent turn; the model writes that turn 8 times, and a reward scores each
attempt.

**Our first run trained only on the teacher's tool calls.** The model learned that calling a tool
was always rewarded, stopped replying to customers, and collapsed: 0.5 of 30 on held-out tasks,
against Arm B's 17.0.

**Our second run followed the loop.** It trained on 177 teacher fixes at real failures, with one
reward per failure type (Tool, Arg and Loop), plus agreement with the teacher's fix. It didn't
collapse, halved loops, and cut repeated calls in telecom from 45% to 7%. But it solved fewer
tasks than Arm B (51% against 60% on dev), and the gate rejected it.

The conversations it lost weren't hallucinations. They were **decisions**. It cancelled airline
reservations nobody asked it to cancel, and in telecom it walked customers through phone settings
but skipped the fix on its own side of the account. A cancellation with a real reservation ID
passes every hallucination check, whether or not it should happen. **Rewarding the absence of
hallucination taught the model to act confidently, not correctly.**

**Our third run added a decision reward.** It checked each change against the changes the task
still expected. It drifted towards calling tools, and a guard we'd built to watch the share of
replies stopped training early. It scored 14.5 of 30, below Arm B's 17.0.

### What we learned

* **Verified data beats instructions.** What healed the agent was hundreds of correct turns, each
  checked by the benchmark's own evaluator.
* **Decide the rule before you see the result.** Every model we trained was judged by a rule
  written down before it ran. Several promising ideas didn't pass, and the rules kept us honest.
* **Fewer hallucinations isn't the same as more success.** The gate caught this twice.
* **The grain of the fix matters.** Rewriting one turn removed most hallucinations but solved no
  more tasks; training on whole verified conversations did both.
* **Hallucinations and decision errors are different problems.** Once the hallucinations were
  gone, the remaining failures were about what to change, and when: in 37 of Arm B's 55 failed dev
  and validation conversations, the classifier finds no hallucination at all. A loop that heals
  hallucinations has nothing left to fix there. That's the next frontier for self-healing agents.

### Reproduce it yourself

Every run saved its results, so you can step through all five experiments on a laptop in seconds,
with no GPU or API keys. Each one is introduced, recomputed from the saved results, and ends with
the commands that rerun it from scratch:

```bash
git clone <repo> && cd <repo>
just reproduce            # or: python scripts/blog/reproduce.py
```

![Terminal output of just reproduce 2. Experiment 2 of 5: Can you just tell it to be careful? Success: Arm A 14.5%, Arm C 13.0%, a difference of −1.5 points (−7.0 to +4.0). Tool or Arg failure: 95.0% and 95.5%. Loop: 49.0% and 49.5%. Result: no clear difference, the 95% range of C − A includes zero. Asking nicely doesn't work.](img/reproduce-output.png)

It ends by checking that every number it recomputed appears in this post exactly as written.

### What's next

The next step is a reward that judges decisions as well as structure, and scores whole
conversations rather than single turns. We'd love to hear how others approach it.

**Everything is public:**

* **Code:** [github.com/CharlieGalvin/lexsi-labs-reflexive-agent](https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent)
* **Arm B, the healed model:** [reagent-lexsi/sft-aya32b-r16-v2](https://huggingface.co/reagent-lexsi/sft-aya32b-r16-v2),
  a LoRA adapter for Aya Expanse 32B. Every other version we trained, adopted or not, is in
  [reagent-lexsi](https://huggingface.co/reagent-lexsi), each with its result and lineage.
* **Data:** [reagent-lexsi/tau2-golden-trajectories](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories),
  the teacher's verified conversations and the hard subset (CC BY-NC 4.0).

### Acknowledgements

Thanks to **Lexsi Labs** for the hackathon and the stack. Two notes from using it:

* **AgentTune:** its `FailureDetector` sees only tool calls, so it misses loops made of repeated
  replies; we added a turn-level monitor for those. `lexsi_provenance.json` worked well as the
  lineage record for every adapter we published.
* **AuditKIT:** its outcome oracle agreed with τ²-bench's own verdicts on all 600 test conversations.
  Its `agent_loop_detection` counts near-identical replies, such as repeated goodbyes, as loops, so it
  couldn't separate the untuned and healed agents (125 and 126 of 200) where our classifier could
  (49% and 27%).
