---
title: "Reflexive Agents: A 32B Agent That Heals Its Own Wrong Actions, from 14.5% to 60%"
thumbnail: /blog/assets/reflexive-agents/social-preview.png
authors:
- user: katiemarieek
- user: cp89
- user: cg3849
---

# Reflexive Agents: A 32B Agent That Heals Its Own Wrong Actions, from 14.5% to 60%

*12 min read · This is the full story. The [first post](https://huggingface.co/blog/cg3849/reflexive-agents) covered the first half.*

Agents are increasingly trusted with write access: cancelling orders, changing flights, editing
accounts. When they go wrong, the result isn't a wrong sentence but a wrong action, and a wrong
action changes a real record. Telling the agent to be careful doesn't stop it.

We asked whether an agent can **heal itself**: find its own failures, learn from verified
corrections, and keep a new version only if it measurably does better. We built that loop for
**Aya Expanse 32B** on [τ²-bench](https://github.com/sierra-research/tau2-bench), with
**GLM-5.3-Flash** as the teacher and the Lexsi Labs stack ([AgentTune](https://github.com/Lexsi-Labs/AgentTune)
and AuditKIT).

**What we found:**

* **Healing fixes the failures the loop can see.** Training on the teacher's verified
  conversations took the agent from **14.5% to 51.5%** of test tasks, and cut conversations with a
  made-up tool or argument from **95% to 10%**. Asking it nicely changed nothing.
* **It can't see wrong decisions.** After healing, most failures were real tools used with real
  values to do the wrong thing. Rules, a consent checker and an LLM judge could all tell a
  conversation had failed, but rarely *where*, so healing where they pointed didn't help.
* **The bottleneck was simpler: the agent hadn't learned its lessons yet.** Training on the same
  verified data for twice as long took it to **60.0%** (+9.9 points, 95% range +3.9 to +16.0), close
  to GPT-4.1's 61.5% on the same tasks.
* **It's safer and cheaper too.** Changes made without the customer's yes fell from 0.152 to 0.010
  per conversation, about 15× fewer. Served in FP8, it needs half the memory for its weights, with
  the same score.

![Task success on the τ²-bench test, 100 tasks. Base Aya through the API 14.5%, with a careful prompt 13.0%, base Aya on our own GPU 11.5%, healed (v2) 50.1%, trained longer 60.0%, trained longer plus its own practice 58.9%. Training longer added 9.9 points over the healed model, 95% range +3.9 to +16.0. Base arms use 2 samples per task, the others 8.](img/test-success-arc.png)

*Figure 1. The whole journey on the 100 test tasks. Grey bars are baselines and controls; blue is the final model.*

### The idea, without the jargon

Picture a new trainee on a customer help desk. They're quick and polite, but under pressure they
bluff: they quote an order number that doesn't exist, use a system they don't have access to, or
press the same broken button again and again. That's an agent **hallucinating**.

Now give the trainee a senior colleague who replays each call that went wrong and writes down
how it should have gone. The trainee practises on those corrections. Before they go back on the
phones, a manager runs them through a set of test calls and signs them off only if they solve
more than before. Spot the mistake, get the correction, practise, pass the check, repeat: that's
**self-healing**.

It worked: the trainee stopped bluffing. But they still made wrong calls: refunding the wrong
order, changing a booking before the customer agreed. The manager could tell a call had gone
badly, but not the moment it went wrong. So we looked at the trainee's own record, and found
they still got half of their *practice* calls wrong. They hadn't failed to get corrections.
They hadn't finished learning them. More practice on the same corrections fixed more than any
new way of spotting mistakes.

In our project the trainee is a 32B open model, the senior colleague is a stronger model, and the
manager is a benchmark with a rulebook written down before every run.

![The self-healing loop, animated. The student agent, Aya Expanse 32B with a LoRA adapter, fails a conversation. 1, Detect: a classifier finds the failing turn and flags an invented argument, email="something@example.com". 2, Fix: the teacher rewrites the turn as "Could you confirm your email?". 3, Retrain: the student trains on 146 verified conversations. 4, Gate: dev success rises from 11% to 60%, so the new adapter is promoted and the student moves from adapter v1 to v2.](img/heal-loop.gif)

*Figure 2. The self-healing loop: detect where the agent went wrong, have the teacher write the correct version, retrain on the corrections, and promote the new version only if it passes the gate.*

### The setup

τ²-bench gives an agent a simulated customer, a policy and real tools over a database, in three
domains: retail, airline and telecom. A conversation passes only if the database (and in telecom,
the customer's phone) ends in exactly the right state. An invented argument doesn't just read
badly; it changes the wrong record.

We froze the splits early (train 142 tasks, dev 36, test 100), and test tasks never appear in
any training data. Every model we tested was judged by a rule written down, with its date,
before it ran. GLM-5.3-Flash, chosen from 12 candidate teachers, produced verified conversations
for 158 of 178 train and dev tasks, for **$11** in total.

## Part 1: healing what the agent makes up

### What a hallucinating agent looks like

Untuned, the agent called tools it didn't have, invented IDs, and retried calls that had already
failed. Our classifier labels three kinds of failure:

![Three failure types. Tool: the agent calls a tool it doesn't have, such as check_apn_settings(), a phone tool only the customer can run. Arg: it invents an argument, a placeholder, made-up or ill-typed value such as email="something@example.com". Loop: it repeats itself, sending the same failed call, the same call or the same reply, such as one failing call sent 9 times.](img/failure-types.png)

*Figure 3. The three kinds of hallucination our classifier looks for.*

On the test split, the untuned agent had a Tool or Arg failure in **95% of conversations**. With a
fixed instruction not to hallucinate, it scored 13.0% against 14.5% without it (−1.5 points, 95%
range −7.0 to +4.0), and still hallucinated in 95.5%. The model knows the rule; it doesn't follow it.

Think you'd catch them? Here are three moments based on our test runs:

<iframe src="https://reagent-lexsi-spot-the-hallucination.static.hf.space" frameborder="0" width="100%" height="820"></iframe>

### Healing works, and whole conversations are what teach

We fine-tuned the agent (QLoRA, rank 16) on the teacher's whole verified conversations for the
train tasks it failed. On the 100 test tasks, it solved **51.5% against 14.5%**: **+37 points**
(95% range +28 to +46). Conversations with a made-up tool or argument fell from 95% to 10%, and
general benchmarks (instruction following, maths, multilingual knowledge) didn't drop.

Here is the same test task, a customer returning a broken office chair, before and after.

**Untuned.** Before the customer has given any details, the agent invents an email and chains
three calls on IDs it doesn't have yet:

```text
Customer: The office chair I ordered showed up with broken pieces. I want to return it.
Agent:    find_user_id_by_email(email="something@example.com")
          get_user_details(user_id="<user_id_from_previous_tool_result>")
          get_order_details(order_id="<latest_order_id_from_user_details>")
Tools:    Error: User not found · Error: User not found · Error: Order not found
```

**Healed.** The agent asks first, then uses only what it has been told or shown:

```text
Customer: The office chair I ordered showed up with broken pieces. I want to return it.
Agent:    I'm sorry to hear that! I'd be happy to help you with the return. First, I need
          to verify your identity. Could you please provide me with the email address
          associated with your account? Or, if you prefer, your first name, last name,
          and zip code.
Customer: I don't remember my email. My name's Mei Davis, zip code 80217.
Agent:    find_user_id_by_name_zip(first_name="Mei", last_name="Davis", zip="80217")
Tools:    mei_davis_8935
```

The loop also runs with no one in it, and its gate works in both directions:

| The loop, run automatically | Fix | Dev tasks (before → after) | Hallucination findings | Gate |
|---|---|---:|---:|---|
| Cycle 1 | one rewritten turn per failure | 4 → 4 | −71% | reject |
| Cycle 2 | whole verified conversations | **6 → 22** | **−87%** | **promote** |

Rewriting the one turn where the agent went wrong removed most hallucinations but solved no more
tasks. **Training on whole verified conversations did both**: the agent has to learn how to finish
the job, not just how not to start it badly.

We also tried reinforcement learning (GRPO) on top. It never beat fine-tuning, but it taught us
something: rewarding the absence of hallucination made the agent act *confidently*, not
*correctly*. It cancelled reservations nobody asked it to cancel. Which brings us to Part 2.

## Part 2: what healing can't see, and what fixed it anyway

### The failures left are decisions

After healing, **37 of the healed agent's 55 failed dev and validation conversations had no
made-up tool, argument or loop at all.** The agent called real tools with real values, then
decided wrongly: it changed the wrong order, acted before the customer agreed, skipped a step or
gave up. More of the same healing data didn't help: two further versions scored the same as the
healed model on validation.

A self-healing loop has to find its own failures: which conversations to heal, and the turn to
correct. For a made-up call, a schema check does both. A wrong decision passes every schema
check, and a deployed agent has no answer key. So we asked: **without the answer key, can a
detector find decision errors, and the turn where they happen?**

### Detectors can find failures, but not where to fix them

We built three: the loop proposing its own new rules from its failure logs (none survived
held-out conversations), a hand-written consent and mismatch checker (did the customer say yes
first, and does the change match the request and the lookups?), and, as the organisers
suggested, a **hybrid judge**: an LLM reading the whole conversation, with code giving it the facts
LLMs get wrong (the changes made, prices, the classifier's findings).

On train tasks, τ²-bench's answer key says which conversations failed and where each first went
wrong, so we could score every detector against it. On 224 held-out conversations:

| Detector | Failures caught | Successes wrongly flagged | Flags on the turn it went wrong |
|---|---:|---:|---:|
| Rules (the Part 1 classifier) | 56% | 14% | 0 of 52 |
| Consent and mismatch checker | 27% | 8% | 15 of 28 |
| Rules + checker | 69% | 20% | 15 of 67 |
| Hybrid judge | 42% | 9% | 11 of 44 |

**No detector did both things the loop needs**: catch most failures, and point to the turn where
they went wrong. The rules catch symptoms early in the conversation; the checker usually finds the
right turn but catches only one failure in four; the judge sits between them. A teammate who
didn't build the checker ran it again on the dev and validation conversations: it flagged 6 of the
37 decision failures.

Acting on the detectors didn't help either. Used as a live guard before each change, the checker
blocked 5 replies in 2 of 60 validation conversations and didn't move the score. Healing from the
turns it flagged gave 21 verified conversations: too few to retrain on.

### First, measure properly

Then we noticed something uncomfortable. **The same model, on the same 30 validation tasks, scored
anywhere from 12 to 19 from one run to the next**, because the simulated customer behaves
differently each time. Our earlier comparisons, at one or two runs per task, couldn't tell a real
gain of two or three tasks from luck. From here on every comparison used 8 runs per task, paired
task by task, with a 95% bootstrap range. Measured that way, the healed model scores 50.1% on test,
not the 51.5% from 2 runs.

### The real bottleneck: the agent hadn't finished learning

The clue was in the healed model itself. **It passed only 55 of the 112 train tasks it had been
trained on**, each with a verified conversation in its data, and its held-out loss was still falling
when training stopped. So we changed one thing: 4 epochs instead of 2, same data, same settings.

| Change from the healed model | Train tasks (112 × 1) | Validation (30 × 8) | **Test (100 × 8)** |
|---|---:|---:|---:|
| Retail | −2 | −0.9 | 0.0 |
| Airline | +3 | +1.4 | +0.6 |
| Telecom | +4 | +2.0 | **+9.3** |
| **Total** | **55 → 60** | **16.0 → 18.5** | **50.1% → 60.0%** |

**On test: +9.9 points (95% range +3.9 to +16.0).** The gain has the same shape on all three sets
of tasks, largest in telecom, with its long multi-step troubleshooting. It is larger on unseen test
tasks than on the training tasks themselves, so training longer improved how the agent
generalises, not just what it remembers. What it fixed, mostly, was wrong changes: failed
validation conversations with an unexpected change fell from 65 to 52.

We're open about how this result was protected. On validation the gain (+2.5 of 30) had a range
that included zero, so it went to test as a declared extra arm, labelled as such. And after a
borderline first test at 2 runs per task (+8.0, range −1.0 to +16.5) we extended both models to 8:
a decision made after seeing a result. The 6 new runs alone give +10.5 (+4.0 to +17.0).
<!-- TODO (Thu): the second training seed, judged by its pre-registered rule (STAGE2.md, Stage 2d). -->

### Does the agent's own practice add anything? No

We also tried the purest form of self-healing: let the healed agent attempt each train task twice,
and train on what τ²-bench verifies. Of its 224 conversations, 108 passed and 100 were kept,
covering 74 tasks. Added to the teacher's data and trained the same longer way, it scored 58.9%
against 60.0% (−1.1 points, 95% range −6.6 to +4.4). **What mattered was how long the agent trained,
not whose conversations it trained on.**

### Safer, and cheaper to run

**The final model is more careful, not less.** Changes made without the customer's explicit yes
fell from 0.152 to 0.010 per conversation (95% range of the change −0.310 to −0.035): 6 of its 800
test conversations had one.

**It's close to GPT-4.1 for a fraction of the cost.** On the same 100 test tasks, GPT-4.1 as the
agent solved 61.5% (2 runs per task), at $0.235 per conversation. The healed Aya, on one H100 of our
own, costs $0.052 per conversation; the final model is served the same way.
<!-- TODO (Katie, today): the final model's own cost per conversation and per 1,000 replies. -->

**And it fits on less hardware.** Served in FP8, its weights take 32.2 GiB of GPU memory instead
of 60.4 GiB. The memory freed doubles the context, from 32K to 64K tokens, and its validation
score doesn't move (18.5 against 18.5). Serving the model ourselves rather than through Cohere's
API made no difference within noise (11.5% against 14.5% for the base model, range −8.0 to +1.5).
<!-- TODO (Katie, Thu): retention on the final model. -->

### What we learned

* **Verified data beats instructions.** What healed the agent was correct conversations, each
  checked by the benchmark's own evaluator.
* **Whole conversations teach; single turns don't.** The gate caught this, with no one in the loop.
* **A self-healing loop can only heal what it can locate.** Made-up calls are easy to find. Wrong
  decisions are not, and better detectors didn't fix that.
* **Check whether the model has learned what you already gave it.** Our agent failed half the
  tasks it had trained on. Training longer did more than any new detector, and more than its own
  practice.
* **Measure with enough runs, and decide the rule first.** One or two runs per task hid a swing of 7 tasks in 30.
  Every result here was judged by a rule written down before it ran, including the ones that failed.

### Limitations

* **One model, one benchmark.** Every result is Aya Expanse 32B on τ²-bench's three
  customer-service domains, with 100 test tasks.
* **One training run behind the final gain**, until the second seed is in. <!-- TODO: update Thu. -->
* **Two disclosed departures** for the final model: it went to test without meeting our validation
  rule, and its test was extended after a borderline first result.
* **GPT-4.1 is a reference, not a paired comparison**: 2 runs per task against our 8.
* **It depends on a stronger teacher**, and on conversations the benchmark can verify.
* **Non-commercial.** The adapters inherit Aya Expanse 32B's CC BY-NC 4.0 licence.

### Everything is public

* **Code, and the dated rulebook for every run:** [github.com/CharlieGalvin/lexsi-labs-reflexive-agent](https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent)
  (`docs/experiments/STAGE2.md`)
* **The final model:** <!-- TODO: link, e.g. reagent-lexsi/sft-aya32b-v2-e4 --> a LoRA adapter for
  Aya Expanse 32B. The healed model from Part 1 is
  [reagent-lexsi/sft-aya32b-r16-v2](https://huggingface.co/reagent-lexsi/sft-aya32b-r16-v2), and every
  other version we trained, adopted or not, is in [reagent-lexsi](https://huggingface.co/reagent-lexsi).
* **Data:** [reagent-lexsi/tau2-golden-trajectories](https://huggingface.co/datasets/reagent-lexsi/tau2-golden-trajectories),
  the teacher's verified conversations (CC BY-NC 4.0). <!-- TODO: Stage 2 conversations and practice data. -->

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("CohereLabs/aya-expanse-32b", device_map="auto")
model = PeftModel.from_pretrained(base, "reagent-lexsi/sft-aya32b-v2-e4")  # TODO: confirm the repo name
```

### Reproduce it yourself

Every run saved its results, so you can recompute the experiments on a laptop, with no GPU or API
keys:

```bash
git clone https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent && cd lexsi-labs-reflexive-agent
just reproduce            # Part 1's experiments
python scripts/arms/stage2b_compare.py S2 S2E4 --split test --samples 8   # the +9.9
```
<!-- TODO: add Part 2 to `just reproduce`, so it checks this post's numbers too. -->

### Acknowledgements

Thanks to **Lexsi Labs** for the hackathon and the stack. AgentTune's `lexsi_provenance.json`
recorded the lineage of every adapter we published; its `FailureDetector` sees only tool calls, so
we added a turn-level loop monitor. AuditKIT's outcome oracle agreed with τ²-bench's own verdicts on
all 600 test conversations; its loop detector counts near-identical replies, such as repeated
goodbyes, as loops, so we relied on our own classifier there.

### Citation

```bibtex
@software{reflexive_agents_2026,
  title  = {Reflexive Agents: a self-healing loop for AI agents' wrong actions},
  author = {Kitchen, Katie and Picardi, Chiara and Galvin, Charlie},
  year   = {2026},
  url    = {https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent}
}
```
