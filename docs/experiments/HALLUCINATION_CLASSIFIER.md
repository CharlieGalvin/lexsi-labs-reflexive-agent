# Tool / Arg / Loop classifier (4 Nov 2026, SEL-12)

**What it is:** `reflexive.hallucination.classify_hallucination` reads a conversation as the
agent saw it and lists every structural failure in it, typed **TOOL**, **ARG** or **LOOP**. The
primary failure is the earliest one. It's rule-based and needs no τ²-bench install or API calls:
tool schemas come from `configs/tool_schemas.json` (`just tool-schemas`).

> **Update 7 Nov (SEL-62): the hand-off exemption is removed.** It relied on one dataset's tool
> name (`transfer_to_human_agents`), and it hid A56 (Arm A telecom), a conversation that ran to
> the step limit swapping goodbyes. Chat after a hand-off now counts as `repeated_message` /
> `step_limit` like any other. On the 148 Arm A and C telecom conversations this changes only
> A56 (it gains `step_limit` and `repeated_message`; its primary failure stays TOOL). The
> precision figures below were measured **with** the exemption, and the "0 false alarms on the
> 158 golden records" hasn't been re-checked without it (the golden files weren't to hand).
> The same loops are now also caught while they run, by `reflexive.monitor` (`no_progress`).
> Tools can also come from the live request (`classify_record(rec, tools=...)`), with
> customer-only names as an optional label.

| Type | Kind | What it means |
|---|---|---|
| TOOL | `customer_tool` | Called one of the customer's phone tools (telecom), which the agent doesn't have |
| TOOL | `unknown_tool` | Called a tool that doesn't exist at all |
| ARG | `placeholder` | Put a description where a value goes: `"customer_id from previous tool result"`, `"YYYY-MM-DD"` |
| ARG | `ungrounded` | An ID, name, phone number, email or date of birth not mentioned anywhere earlier: invented |
| ARG | `unknown_param`, `missing_param`, `wrong_type` | The call doesn't fit the tool's schema |
| ARG | `malformed` | Arguments that aren't JSON, or an `Action:` block the parser couldn't read |
| LOOP | `repeated_failed_call` | The same call again after it already failed |
| LOOP | `repeated_call` | The same call 3 times |
| LOOP | `repeated_message` | The same reply 3 times (since 7 Nov also after a hand-off: see the update below) |
| LOOP | `step_limit` | τ²-bench stopped the conversation at 200 steps (since 7 Nov also after a hand-off) |

## On Aya's telecom rollouts (Arm A and Arm C, 74 train tasks each)

| | Arm A (baseline) | Arm C (prompt-only) |
|---|---:|---:|
| Failed conversations | 69 | 72 |
| … with a TOOL failure | 69 (100%) | 72 (100%) |
| … with an ARG failure | 60 (87%) | 60 (83%) |
| … with a LOOP failure | 50 (72%) | 52 (72%) |
| Primary failure ARG / TOOL | 51 / 18 | 54 / 18 |
| Failed with no structural failure | 0 | 0 |

**Every failed telecom conversation calls the customer's phone tools.** The typical opening
turn is one batch: look up `"customer's phone number"` (a placeholder, so ARG comes first), then
call `can_send_mms`, `check_apn_settings` and others that only the customer can run. Then it
retries the same batch (LOOP). Arm C's prompt changes none of this.

**Why it calls them:** these names aren't invented. Telecom's agent policy includes τ²-bench's
`tech_support_manual.md`, which names the customer's tools in call syntax
(`check_network_status()` 4 times, `check_apn_settings` 3, …) so the agent can *ask the customer*
to run them. Aya reads a tool name in its own instructions and calls it. That's a confusion about
whose tools they are, which is why Arm C's "never invent a tool" has no effect.

Placeholders are much more common than invented values: 59 conversations against 3 in Arm A.
The invented values seen are made-up IDs (`C00123456789`, `L0012345`) and a year added to a
birthday the customer gave without one.

## Spot check: precision per type

10 findings of each type, from 30 different failed conversations, spread across kinds, with a
fixed seed. Each was judged by reading the conversation. Labels and notes:
[`hallucination_labels.json`](hallucination_labels.json); re-run with `just hallucination-spot-check`.

| Type | Correct | Precision (95% CI) | Kinds checked |
|---|---:|---:|---|
| TOOL | 10 of 10 | 100% (72–100%) | customer_tool 8, unknown_tool 2 |
| ARG | 10 of 10 | 100% (72–100%) | placeholder 5, ungrounded 3, malformed 2 |
| LOOP | 10 of 10 | 100% (72–100%) | repeated_call 3, repeated_failed_call 3, repeated_message 2, step_limit 2 |

The spot check changed the classifier twice before these numbers:
- **A false positive:** a conversation where the agent really did hand over to a human, then
  kept saying goodbye until the step limit. Chat after a successful hand-off no longer counts
  as a loop. *(Reversed on 7 Nov: see the update below.)*
- **Misses found while reading:** placeholders written with underscores or CamelCase
  (`customer_id_from_search`, `CustomerIDfromPreviousToolResult`), `"user's full name"`, and
  `19XX-05-14`. These are now caught.

## Limitations

- **Telecom only, one student; retail and airline only partly tested.** Only Aya's telecom
  rollouts exist so far. Retail and airline are covered by built test cases, not by real
  transcripts.
  - TOOL, the schema checks, placeholders and LOOP don't depend on the domain, so they should
    carry over.
  - The risk is `ungrounded`: retail and airline IDs (orders, items, products, payment
    methods, flights) all come from tool results, and no real retail or airline conversation
    has been checked.
  - Replaying τ²-bench's correct actions doesn't settle it. The gold action lists leave out
    the lookups (airline task 14 is only `cancel_reservation` + `book_reservation`), so IDs
    look invented in the replay although a real agent would have looked them up first.
  - **The decisive test:** the golden v1 retail and airline records (clean teacher passes,
    so any `ungrounded` finding is a false positive) and `results/fixes/aya.jsonl` (real
    student failures). Both are only on Charlie's laptop for now. Re-run the spot check when
    Arm A covers retail and airline too.
- **Fixed while checking retail:** every τ²-bench retail and airline customer has an
  `@example.com` email, and "example" was a placeholder word, so every email lookup was
  flagged. Now only "for example" counts, with a regression test.
- **Precision only.** Recall (failures it misses) isn't measured. Reading the 30 samples turned
  up the misses listed above.
- **Small sample.** 10 per type gives a 95% lower bound of 72%.
- **Labelled once, by Claude.** The labels need a person to confirm them.
- **`unknown_param`, `missing_param` and `wrong_type` never occurred** in these rollouts. Only
  the built tests exercise them.
- **Not judged:** wrong reasoning or policy mistakes with well-formed calls. Those come out as
  `NONE`.
