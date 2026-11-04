# Student review: where Aya Expanse 32B goes wrong (4 Nov 2026, SEL-7, SEL-35)

**Question:** before training, what does the student get wrong, and is any of it the *format*
rather than the *behaviour*? The answer decided the data contract: keep Cohere's `Action:`
format, and train on behaviour.

**Answer:** Aya passed **7 of 30** train tasks (retail 1/10, airline 5/10, telecom 1/10). The
format works: **0 unreadable tool calls in 266 replies**. Every failure is behaviour:

- calling tools it doesn't have;
- inventing IDs;
- retrying calls that already failed;
- batching dependent calls together.

These are exactly what the golden trajectories show done right.

**Browse every conversation:** [`reports/aya_student_review.html`](reports/aya_student_review.html).
GitHub shows HTML as source code, so download it, or open it from your clone in a browser.
Regenerate it with `just student-review` (needs `results/fixes/aya.jsonl`).

## Setup

- **Student:** Aya Expanse 32B (`c4ai-aya-expanse-32b`, Cohere API), prompted in Cohere's
  `Action:` format: the same prompt as the Arm A baseline.
- **Tasks:** 10 random train tasks per domain (seed 0); dev and test excluded.
- **Customer:** GPT-4.1. **Scoring:** τ²-bench's evaluator, judges on.
- **Cost:** $1.54.

## Main reason for each failure

Each of the 23 failures gets one main reason, from the first problem in this order:

| Main reason | Failures | Share |
|---|---:|---:|
| Unknown tool (called a tool the agent doesn't have) | 8 | 35% |
| Invented ID | 5 | 22% |
| Repeated a failed call | 3 | 13% |
| No change made | 2 | 9% |
| Wrong change | 2 | 9% |
| Handed to a human | 1 | 4% |
| Told the customer something wrong | 1 | 4% |

Conversations showing each problem (a conversation can show several): invented ID 10 of 30,
repeated failed call 10, handed to a human 10, unknown tool 9, device not fixed 8 (of the 10
telecom conversations), wrong change 7, no change 6. Half the conversations (15) also wrote a
tool call inside prose, but that's a note, not a failure: the call was still read.

## The four behaviours

1. **Calling the customer's phone tools itself (telecom).** In telecom the customer has 30
   phone tools and the agent has 13. Aya kept calling the customer's:
   - `check_network_status` 15 times;
   - `check_apn_settings` 12;
   - `check_wifi_calling_status` 7;
   - `check_app_permissions` 6;
   - `run_speed_test` 5.

   **55 of its 56 unknown-tool calls** were the customer's phone tools (the 8 commonest are listed
   above). The teacher instead *asks the customer* to check.
2. **Batching dependent calls.** **50% of Aya's tool turns** (54 of 107) make several calls at
   once, often using a placeholder for a value the first call hasn't returned yet. GLM-5.3-Flash
   does it in **4%** (54 of 1,410) of its turns, and only for independent lookups.
3. **Inventing IDs.** Examples: `#W0000000`, `1234567890`, `sara_doe_496`. Lookups came back
   "not found" 56 times across 13 tools, usually after a guessed ID. The commonest:
   - `get_user_details` 15;
   - `get_customer_by_phone` 9;
   - `get_order_details` 8;
   - `find_user_id_by_email` 7.
4. **Retrying the same failed call** unchanged, in 10 of 30 conversations: the Loop failure type.

## What the agent sees (the check behind the contract's "agent's view" rule)

In 3 of the 10 telecom conversations the customer took phone actions. The agent was shown
them **0 times**. So training uses the conversation as the agent saw it, not the full saved
log. The golden trajectory build checks its rebuilt view against these 30 conversations: 30/30
exact matches.

## What it means

- **The format isn't the problem,** so the contract keeps Cohere's `Action:` format (rule 1).
- **The behaviours map onto the failure types:**
  - unknown tools and the customer's phone tools are **Tool** failures;
  - invented IDs are **Arg** failures;
  - retried failed calls are **Loop** failures.

  These are the hard subset's labels (SEL-7).
- **30 conversations is a sample.** Proportions like "35% unknown tool" are rough (±15 points
  or so). The full Arm A baseline runs on the test split, at the end.
