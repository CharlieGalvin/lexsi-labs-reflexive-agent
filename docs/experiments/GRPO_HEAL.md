# Heal GRPO (arm H): training on the teacher's fixes at Tool / Arg / Loop failures (SEL-75)

**Status: done, 11–12 Nov.** H beats Arm A but not SFT v2. It fails its validation rule and loses to
v2 on dev success. It halves loops, but makes more wrong *decisions*. SFT v2 stays Arm B.

## 1. Result

| Dev: 36 tasks × 2 samples, oracle-scored | Arm A | SFT v2 (S2) | H |
|---|---:|---:|---:|
| Success, mean of 2 samples | 2/36 (6%) | **21.5/36 (60%)** | 18.5/36 (51%) |
| Conversations with a Tool or Arg failure | — | 5/72 | 5/72 |
| Conversations with a loop | — | 12/72 | **6/72** |
| Classifier findings per conversation | — | 1.17 | **0.25** |

**H against SFT v2, paired by task** (95% range in brackets):

| | Retail | Airline | Telecom |
|---|---:|---:|---:|
| Success (points) | −3.3 (−23 to +20) | −16.7 (−58 to +17) | −10.0 (−23 to +3) |
| Loops | −10.0 | −8.3 | −6.7 |
| Repeated calls | −7.2 | −1.9 | **−38.5 (−59 to −1)** |

* **The gate rejects H** (`just gate-check S2 H`, sample 0): 24 → 19 passed. Only H passed 2 tasks,
  only v2 passed 7 (McNemar p = 0.18).
* **The one clear difference:** telecom's repeated calls, 45% → 7%. No success difference is clear
  on its own, but H is lower in every domain.
* **Against Arm A:** far above (18.5 against 2 of 36). *The paired comparison is pending; see §7.*

**Validation** (the rule, §5): H passed at most 17.0 of 30 against a bar of 19.5, so it **fails**.

| Validation (30 tasks) | Retail | Airline | Telecom | Total |
|---|---:|---:|---:|---:|
| SFT v2 | 6.5 | 2.5 | 8.0 | **17.0** |
| H | 5.5 | 2.5 | ≥ 7.0 (4 crashed) | **≤ 17.0** |

A serving check passed first: v2 scored 7 of 13 on retail on H's server, against its earlier 6.5.

## 2. Why H lost to v2: decision errors, not hallucinations

The hallucination rewards worked: Tool and Arg failures stayed rare, and error spirals and loops
fell. But the rewards check *how* a call is made, not *whether* it should be made. So the
conversations H lost are wrong decisions:

* **Airline and retail: acting where it shouldn't.** v2's unrequested calls were error spirals:
  one failing call retried 9 and 7 times until τ²-bench stopped the conversation. H has none.
  Instead, H makes valid but unrequested changes:
  * 12 airline cancellations (task 39: the 3 asked for plus 4 more);
  * an unrequested refund certificate;
  * a retail order cancelled instead of modified.

  Missing changes (11 v2, 10 H) and wrong details (10 each) didn't change.
* **Telecom: fixing the wrong side.** H gets the customer to do *more* on the phone (19.5 actions
  per conversation, v2 16.6) but skips **its own account fix** more often: 7 missed agent actions
  against 2, mostly `enable_roaming`. In 3 of the 4 telecom conversations H lost, an account action
  was missing. All 47 Tool fixes taught "don't call the customer's tools; tell them what to do",
  and H over-learned it.

**The lesson:** a cancellation with a real ID passes the Tool, Arg and Loop rewards, and Chiara's
classifier, whether or not it should happen. A heal reward needs a **decision check**, such as
rewarding only changes the task still expects (as SEL-73 does for its fix points).

How we found this: `scripts/arms/failure_diff.py` for retail and airline, and `just arm-why S2 H
--domain telecom` for telecom. The numbers are small (airline has 6 tasks), so treat this as a
direction.

## 3. What H is

The README's loop: **classify the student's failures, have the teacher fix them, retrain.**

1. **Heal points.** Chiara's classifier labels base Aya's (Arm A's) failed conversations on the 112
   healable train tasks. The first **Tool**, **Arg** and **Loop** failure of each conversation is a
   point: **178 points** (TOOL 47, ARG 103, LOOP 63).
2. **Teacher fixes.** GLM-5.3-Flash rewrites the failing turn (`PrefixTeacher` + `check_fix`, as in
   SEL-47): **177 of 178 accepted**. A fix is a tool call or a reply, often asking the customer for
   a missing ID.
3. **Training prompts.** Each fix becomes one prompt: the conversation up to the failure, with the
   fix as the reference turn. Half as many **anchor** turns from the teacher's own conversations
   keep ordinary turns right. 10% of tasks are held out.
4. **Training.** GRPO from SFT v2's adapter, with G's settings (100 steps, 8 samples, learning rate
   1e-5, KL weight 0.04), so H differs from G only in data and reward.

**Rewards** (`reflexive.heal_rewards`): five reward functions, each charted separately in W&B.

| Reward | Weight | Scores 1 when |
|---|---:|---|
| `tool` | 1.0 | every call uses one of the agent's own tools |
| `arg` | 1.0 | no placeholder, invented ID or schema error |
| `loop` | 0.5 | no repeat of a failed call, and no third identical call or reply |
| `fix` | 1.0 | it matches the teacher: for a call, the same tools and arguments; for a reply, the same content words |
| `guard` | 0.5 | the call is the whole reply, and it doesn't give up |

**From Chiara's review:**
* `tool`, `arg` and `loop` only count when the turn is the teacher's kind (reply or call).
  Otherwise a generic reply scored 3.0 of 4.0 where the teacher called a tool, more than a call
  with one invented ID.
* What a reply says is scored, so generic replies score low.

**Safeguards:**
* **`just grpo-heal-reward-check`, on the real fixes:** every teacher turn scores the maximum (4.0);
  every student failed turn fails its own type's reward (TOOL 42/42, ARG 92/92, LOOP 53/53); canned
  replies score 0.5 where the teacher called and 3.2 where it replied.
* **A drift guard:** training stops if the share of replies drifts more than 10 points from the
  teacher's, compared with v2's start. It never fired.

## 4. Training and held-out check

* **Precheck:** v2 had something to learn on 88% of prompts (G's precheck: 32%).
* **Training:** 100 steps at 2.21 min/step, peak 64.2 GB, commit `b82e182`. Adapter
  `reagent-lexsi/grpo-heal-aya32b-v2`. In the samples:
  * Tool failures fell to 0 after step 40;
  * Arg failures halved (11% → 6%);
  * agreement with the teacher's fixes rose (0.60 → 0.72).
* **Held-out check** (30 turns from 11 unseen tasks): reward +0.11 over v2 (95% range −0.06 to
  +0.30). Every failure type improved; ordinary turns didn't change, so there's no sign of G's
  collapse.

## 5. The validation rule (fixed before any result)

The same rule as G and SFT v3/v4. H beats v2 on the 30 validation tasks (2 samples each) only if:

1. its total is **19.5 or more**. v2's samples scored 16 and 18 (mean 17.0, spread 2), so the bar
   is above 19.0, and 19.5 is the next possible mean;
2. **no domain drops by more than 1 task**: retail ≥ 5.5, airline ≥ 1.5, telecom ≥ 7.0;
3. **Tool/Arg failures are no higher than v2's:** at most 4 of 60 conversations.

Scored as v2's 17.0 was: the runs' own verdicts, with Tool/Arg counted by the classifier.

## 6. Assumptions

1. **The failures are base Aya's,** because v2 rarely makes Tool or Arg errors. So the prompts show
   base Aya's earlier turns, not v2's.
2. **One point per failure type per conversation:** its first occurrence.
3. **A fix is accepted by `check_fix`** (no Tool, Arg or Loop failure, no giving up), not by
   re-running the task.
4. **Every failure reward applies to every prompt.** The labels set the data mix; the weights set
   the focus (Tool and Arg at 1.0, Loop at 0.5, since the runtime monitor handles loops).
5. **Replies are scored by content-word overlap,** not an LLM judge. That's cheap, but it can't tell
   two equally good replies apart.
6. **No validation, dev or test task is ever used for training.**

## 7. Reproduce, and what's open

```bash
just heal-fixes && just grpo-heal-data && just grpo-heal-reward-check          # no GPU; OPENROUTER_API_KEY
just grpo-train configs/train/grpo_heal_aya32b.yaml --push                      # GPU pod, ~4 h
just grpo-heldout configs/train/grpo_heal_aya32b.yaml
ADAPTER= EXTRA_ADAPTERS="heal=runs/grpo-heal-aya32b-v2" just sft-serve          # then, in another terminal:
just arm-val H configs/sft/arms.grpo_heal.yaml && just arm-samples S2 H --split val
for d in retail airline telecom; do just rollout H $d --split dev --samples 2 --config configs/sft/arms.grpo_heal.yaml; done
just arm-compare results/episodes/arms/S2-retail-dev.jsonl results/episodes/arms/H-retail-dev.jsonl   # and each domain
just gate-check S2 H && just arm-why S2 H --domain telecom
```

**Open:**
* **The paired comparison with Arm A:** Arm A's dev files arrived on 12 Nov.
* **4 crashed telecom validation conversations,** and H's Tool/Arg count on validation. Neither can
  change the decision.

**Next steps the result points to:**
* add a decision check to the reward;
* train from base Aya, where Tool and Arg failures are common;
* score whole conversations, not single turns.
