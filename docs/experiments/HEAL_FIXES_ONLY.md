# The heal loop's teacher half on real failures (SEL-47)

**Status:** final for this sample, 7 Nov. One `--fixes-only` cycle on 9 train tasks. **Nothing
was trained or gated:** the student behind Cohere's API can't load an adapter (needs SEL-63).
**Owner:** Chiara. **Metrics:** [`results/heal-sample3-judged.metrics.json`](../../results/heal-sample3-judged.metrics.json).

## Question

When Aya fails a τ²-bench task, can the teacher fix it **at runtime**, from the conversation as
the student saw it up to the failing turn, without re-running the task? And what would a first
heal cycle train on?

## Setup

| | |
|---|---|
| Command | `just heal --fixes-only --sample 3 --out results/heal/sample3-judged` @ `97ae092` |
| Tasks | 3 random **train** tasks per domain (seed 0): retail 28, 47, 88; airline 0, 23, 33; telecom 2 mobile-data, 1 service |
| Student | Arm A: Aya Expanse 32B via Cohere, `Action:` format, temperature 0 |
| Customer | GPT-4.1 (τ²-bench's default) |
| Scoring | τ²-bench's evaluator, **judges on** (retail NL assertions, airline communicate) |
| Teacher | GLM-5.3-Flash (`heal.PrefixTeacher`): sees the agent's view up to the classifier's first finding, rewrites that turn; up to 3 attempts |
| Fix check | `heal.check_fix`: not empty, not the failed turn again, and the classifier finds **nothing in the new turn** |

## Results

| | |
|---|---|
| Passed | **1/9** (95% 2–44%): retail 88 only |
| Failures | 8: ARG 4, TOOL 2, **NONE 2** (no finding) |
| Had a turn to fix | 6 of 8 |
| Fixed | **6 of 6**, all on the first attempt |
| Turn fixed | turn 3 every time: the student's **first** action |
| Cost | $0.58: student $0.30, customer $0.27, teacher $0.005 (judge calls not priced) |

| Domain | Task | Checks | Failure | First finding (turn 3) | Fixed |
|---|---|---|---|---|---|
| retail | 28 | DB 0, NL 0 | ARG | `find_user_id_by_email(email='something@example.com')`: never mentioned | ✅ asks for the email |
| retail | 47 | DB 0, NL 1 | ARG | same invented email | ✅ asks for the email |
| airline | 0 | DB 0, communicate 1 | NONE | (nothing found) | ⛔ no turn to fix |
| airline | 23 | DB 0, communicate 0 | ARG | `get_user_details(user_id='sara_doe_496')`: never mentioned | ✅ asks for the user ID |
| airline | 33 | DB 0, communicate 1 | NONE | (nothing found) | ⛔ no turn to fix |
| telecom | mobile data (Hard) | env 0 | TOOL | `check_network_status`: the customer's phone tool | ✅ asks to verify identity |
| telecom | mobile data (Hard) | env 0 | TOOL | `check_network_status`: the customer's phone tool | ✅ asks to verify identity |
| telecom | service (Easy) | env 1, action 0 | ARG | `get_customer_by_phone(phone_number="customer's phone number")` | ✅ asks to verify identity |

## What it shows

1. **Runtime fixing works, cheaply.** Every failure with a finding was fixed on the first try,
   for about $0.001 each, without re-running anything. (The two earlier smoke runs agree: every
   failure with a finding, 5 in each, was fixed on the first try.)
2. **But a first cycle teaches one lesson: identify the customer first.** Every fixable failure
   is the student's first action, looking someone up with an identity it doesn't have (an
   invented email or user ID, a placeholder phone number, the customer's own phone tools).
   Every fix asks for it. It is the right fix and the student's most common mistake, but it
   is the only one this cycle can see.
3. **Nothing after turn 3 is examined.** The fix is at the *first* finding, and the rest of
   the conversation diverges from it, so later mistakes only surface once a trained student
   gets past identification. **Several cycles (SEL-23) are where this approach has to show
   gains;** one cycle can at best fix the opening.
4. **A quarter of the failures can't be fixed this way.** Airline 0 and 33 failed the database
   check with no Tool / Arg / Loop finding: valid calls that did the wrong thing (policy or
   action errors). They have no turn to fix. The teacher's full re-run fixes (SEL-45,
   `scripts/teacher/fixes.py`) can cover them offline.
5. **The near-duplicate cap (`per_lesson=2`) drops nothing here:** 4 lessons, at most 2 each.
   Lessons are per domain and tool, so "identify first" still counts as 4 and dominates the
   training data.

## Caveats

- 9 tasks: the pass rate's range is wide (2–44%), and the 1/9 isn't comparable to the earlier
  unjudged smoke runs (different tasks).
- The fix check is structural: a fixed turn is not a Tool / Arg / Loop failure, not "the task
  would now pass". Whether the trained adapter does better is the gate's job (`DevGate`, all
  36 dev tasks).

## Next

- The first full cycle, once the student is served by vLLM (SEL-63) and the SFT command lands
  (SEL-65).
- Several cycles (SEL-23), to see whether later mistakes surface and get fixed.
- Decide whether NONE failures get offline fixes in the same training set.
