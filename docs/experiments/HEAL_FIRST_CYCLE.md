# The first real heal cycle: rollout → fix → retrain → gate (SEL-68)

**Status:** final for this cycle, 13 Nov. One full cycle on the self-hosted student, from the
**base model** (Arm P), gated on all 36 dev tasks. **The gate rejected the healed adapter**, so
the base model stays. **Run by:** Charlie, on Chiara's SEL-68 code (`44c6af8`).
**Metrics:** [`results/heal-first-cycle.metrics.json`](../../results/heal-first-cycle.metrics.json).

## Question

Does one heal cycle make the student better on tasks it hasn't seen? The cycle runs the student
on train tasks, has the teacher fix each failure at the failing turn, trains an adapter on those
fixes, and lets the agreed DevGate decide whether it replaces the current one.

## Setup

| | |
|---|---|
| Command | `python scripts/heal/cycle.py --config configs/sft/arms.pod.yaml --arm P --sample 10 --out results/heal/first-cycle` @ `44c6af8` (SEL-68 branch) |
| Student | Aya Expanse 32B, base model (no adapter), vLLM on a RunPod H100 with runtime LoRA |
| Heal tasks | 10 random **train** tasks per domain (seed 0): 30 conversations |
| Customer | GPT-4.1 (τ²-bench's default) |
| Teacher | GLM-5.3-Flash via OpenRouter (`heal.PrefixTeacher`), up to 3 attempts per failure |
| Training | `configs/train/sft_heal.yaml`: SFT v1's settings (QLoRA r16, 2 epochs, lr 1e-4), gradient accumulation 2; through the pod heal service (`scripts/pod/heal_service.py`, `train_and_serve.sh`) |
| Gate | `DevGate`, `configs/gate.yaml`: both adapters on all **36 dev tasks**, 1 sample each, temperature 0, oracle-scored with judges on |
| Where it ran | The cycle ran **on the pod itself**, against vLLM and the heal service on `localhost` (the pod's proxy didn't expose ports 8000/8001) |

## Results

**Heal step (train tasks)**

| | |
|---|---|
| Passed | **4/30** (13%; 95% 5–30%) |
| Failures | 26: ARG 19, TOOL 4, NONE 2, LOOP 1 |
| Had a turn to fix | 24 of 26 |
| Fixed | **24 of 24**, all on the first attempt |
| Turn fixed | turn 3 in 21 of 24 (the student's first action); turns 7, 9 and 16 once each |
| Trained on | **16** of the 24: at most 2 fixes per lesson (`per_lesson: 2`), so 8 duplicates of the same lesson were dropped |
| Training | 16 conversations, 938 target tokens (1% of 90,161), 16 steps, mean train loss 0.479; no held-out set |
| Cost | $2.72 (student, customer and teacher; judge calls not priced), 102 conversations |

The lessons, as the classifier names them (count of fixes):

| Domain | Lesson | Fixes |
|---|---|---|
| telecom | ARG/placeholder `get_customer_by_phone` → ask the customer | 6 |
| airline | ARG/ungrounded `get_user_details` → ask the customer | 5 |
| retail | ARG/ungrounded `find_user_id_by_email` → ask the customer | 3 |
| telecom | TOOL/customer tool (`check_apn_settings`, `check_network_status`, `run_speed_test`) → ask the customer | 4 |
| retail | ARG/ungrounded or placeholder (`get_order_details`, `get_user_details`, `modify_pending_order_items`) → ask | 4 |
| airline | LOOP/repeated `get_reservation_details`; ARG/unknown parameter in `search_direct_flight` | 2 |

**Gate (36 dev tasks)**

| | Base (current) | Healed adapter (candidate) |
|---|---:|---:|
| Tasks passed | 4/36 (11.1%) | 4/36 (11.1%) |
| Passed only by this one | 3 | 3 |
| Findings per conversation: TOOL | 2.81 | **0.39** |
| Findings per conversation: ARG | 3.47 | **1.31** |
| Findings per conversation: LOOP | 1.89 | **0.69** |
| Findings per conversation: total | 8.17 | **2.39** (−71%) |
| Domains, net tasks | | airline 0, retail −1, telecom +1 |

**Decision: reject** at check 2 (pass rate): net gain +0 tasks, against the +2 required. McNemar
p = 1.0; 95% bootstrap range of the gain −13.9 to +13.9 points. The checks run in order and the
first that fails decides, so the hallucination check (3), which the candidate would have passed
easily, wasn't reached. The adapter was **not swapped**; the audit record shows `verdict: reject`,
`swapped: false`. The decision counts towards the gate's budget of `max_cycles: 3`
(`results/heal/gate_history.jsonl`).

## What it shows

1. **The loop works end to end on the self-hosted student.** Rollout, classification, teacher
   fixes, training on the pod's GPU, serving the new adapter and the gate all ran, and the gate
   made a recorded decision.
2. **Healing removed most of what it was aimed at.** The fixes were almost all "don't invent an
   identity: ask for it" and "don't call the customer's phone tools". On dev, the healed adapter
   made about 70% fewer Tool / Arg / Loop findings per conversation (TOOL −86%, ARG −62%,
   LOOP −63%). These are the classifier's findings, not a verified hallucination rate.
3. **But it didn't make the student solve more tasks.** 4 against 4, with 3 tasks won and 3
   lost: no difference that 36 tasks can detect. With the opening fixed, the base model still
   fails later in the conversation, which a first cycle can't see (it fixes the *first* finding
   only; [page 6](HEAL_FIXES_ONLY.md) predicted this).
4. **The gate did what it was agreed to do:** it kept the current model when the candidate wasn't
   better at the task.
5. **Compare with SFT v1** ([page 8](SFT.md)), which is the same training on more and different
   data: 112 golden fixes, with loss on every agent turn of a solved conversation, took dev from
   11% to 38%. This cycle trained on 16 fixes and 938 target tokens, one turn each. The healing
   signal here is small; several cycles (SEL-23) or more fixes per cycle are where it has to show
   gains.

## Caveats

- One cycle, one sample per dev task, and a weak base model (11% on dev, as P's 11% on 8 Nov):
  the success comparison rests on 6 discordant tasks.
- The heal step's numbers come from the third attempt at this cycle (below); with the same seed it
  ran the same 30 tasks, but the conversations differ between attempts (the first saw 2/30 pass).
- The training settings for this little data were a guess in `sft_heal.yaml`; 16 steps and a
  finite loss say it trained, not that the settings are right.

## Problems on the way, and what was changed to get the run through

The cycle took three attempts on 13 Nov. Nothing in the analysis changed between them.

| Attempt | What happened | Fix |
|---|---|---|
| 1 (04:34) | Heal rollouts ran, but **all 28 teacher calls failed: OpenRouter 403**, "Key limit exceeded ($50 of $50)". The cycle stopped: "the teacher fixed none of the failures". Kept as `results/heal/first-cycle-403-keylimit` | Raised the key's limit |
| 2 (04:54) | Teacher fixed 12; **training failed: "Disk quota exceeded"**. `train_sft.py` started downloading the 32B model a second time, into the heal service's `HF_HOME` (`/workspace/.cache/huggingface`), while vLLM uses `/workspace/hf_cache` | Restarted the heal service with `HF_HOME=/workspace/hf_cache` |
| 3 (05:18) | Trained and served the adapter, but **`train_and_serve.sh` couldn't see vLLM come up**: its readiness check (`curl -sf .../v1/models`) sends no API key, so a vLLM started with `VLLM_API_KEY` answers 401 until the 20-minute timeout | Put the key in a temporary `/root/.curlrc`, removed straight after; the job finished `done` and the gate ran |

Also on this pod: nginx (the template's "vscode server" proxy) holds port 8001, so the heal
service ran on 8003 and nginx's 8001 block was pointed at it. Two fixes for the SEL-68 branch:

1. `train_and_serve.sh`: send `Authorization: Bearer $VLLM_API_KEY` in the readiness check when it's set.
2. Training should use vLLM's `HF_HOME` (or the heal service should set it), so it never downloads the model again.

## Files

- `results/heal/first-cycle/`: `audit.jsonl` (the cycle's record), `cycle-1.episodes.jsonl` (all 102
  conversations), `cycle-1.fixes.jsonl`, `run.json`, `runs/heal-20261113-052712-c1/` (fixes, training log,
  `run_summary.json`)
- [`results/heal-first-cycle.metrics.json`](../../results/heal-first-cycle.metrics.json): the committed summary
- The adapter `heal-20261113-052712-c1` is on the pod's volume only (not pushed: rejected)
