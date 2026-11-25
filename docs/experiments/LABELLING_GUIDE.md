# Labelling guide: the classifier audit and the 37 (SEL-81)

For SEL-85 (classifier audit) and SEL-87 (the 37). Rules from `docs/experiments/STAGE2.md`;
where this guide and STAGE2.md differ, STAGE2.md wins. Two labellers, Katie and Charlie, label
separately, and neither opens the `-key.csv` files or the other's labels until both have finished.
Agreement is reported as Cohen's kappa (`just label-metrics agreement`).

## What you see

Each item is the whole conversation exactly as the agent and the classifier saw it
(`scripts/labels/make_sheets.py`, seed 0): Turn 0 is the system prompt with the policy, nothing is
cut, and turn numbers are the classifier's own, so the precision sheet's `flagged_turn` points at
the right turn. Model, split, task and the classifier's verdict are hidden.

**Anything in the policy counts as already seen.** The classifier treats a value that appears in
the system prompt (airline's current date, an example ID) as grounded, so do the same: search
Turn 0 before you call a value made up.

## The three hallucination types (as the classifier defines them)

| Type | Kinds | Example |
|---|---|---|
| **Tool**: a call to a tool the agent doesn't have | `customer_tool` (one of the customer's phone tools, e.g. `check_status_bar`; the agent should ask the customer to run it), `unknown_tool` | P01: `check_status_bar({})` → "Tool not found" |
| **Arg**: the right tool, wrong arguments | `placeholder` (a description instead of a value: "customer's phone number", `YYYY-MM-DD`), `ungrounded` (an ID, name, phone, email, zip or date of birth that appears nowhere earlier, policy included), `unknown_param`, `missing_param`, `wrong_type`, `malformed` | `credit_card_2135` built from the last four digits; zip `3019` taken from an email address |
| **Loop**: going round in circles | `repeated_failed_call` (the same call as one that already failed), `repeated_call` (the same call 3 times), `repeated_message` (the same reply 3 times), `step_limit` | P14: `modify_pending_order_items` with the same unavailable item eight times |

Not a hallucination: real tools with real values used to make a wrong choice. That's a decision
error, and it belongs to the 37, not here.

## Sheet 1: classifier precision (P01–P60)

The sheet shows the claimed type, kind and flagged turn. Question: **is the flagged turn really
the claimed kind, as defined above?**

* `yes`: the claimed kind is correct, whatever the conversation's outcome.
* `no`: it isn't. This includes a real hallucination of a different type or kind: record what it
  really is in the notes.
* Leave the label empty and write "unsure" in the notes if you can't decide; we discuss after
  both have finished.

## Sheet 2: classifier recall (R01–R40)

The classifier's verdict is hidden. Question: **does the conversation contain a Tool, Arg or Loop
hallucination anywhere?** Label `yes` or `no`, and list every type you saw in `types_seen`
(e.g. `TOOL ARG`).

Made-up facts in a reply (the agent tells the customer something no tool returned, or says it did
something it didn't) are outside Tool / Arg / Loop: they don't change the yes/no. Note them as
`[made-up facts]` so the write-up can count them.

## Sheet 3: the 37 (D01–D37)

SFT v2's failed dev and validation conversations with no classifier finding. One main error each:
the earliest cause that alone would fail the task. Mention any others in the notes. Sizing only:
never used to build the checker.

| Category | Definition |
|---|---|
| `no_yes` | A state-changing call with no explicit yes from the customer since the agent described that specific change. |
| `nobody_asked` | A change the customer never requested (cancelled, refunded or modified something else). |
| `against_policy` | Did something the written policy forbids, or refused something it allows. |
| `reasoning` | Real tools and values, wrong conclusion: misread a result, wrong item or sum, skipped a step, stopped or transferred too early. |
| `unclear` | No single main error you can name; say why in the notes. |

**What counts as a yes** (same as the "no yes" rule): a clear agreement ("yes", "go ahead",
"please do", "that's fine", "confirm", and equivalents) in the customer's turn after the agent
described that specific change: the action and the order, reservation or line it applies to. A
request up front ("cancel it please") isn't a yes to the agent's later proposal. One yes to a list
of changes the agent set out together covers the whole list; a yes to an earlier proposal doesn't
cover a different change made later.

## Duplicates

22 of the 137 items are word-for-word copies of another item (115 distinct transcripts; e.g. P25,
P32, P40, P55 and P59). Label every item on its own sheet's question. Report with the cluster
bootstrap by task, and say so.

## Afterwards

Each labeller's CSVs (`results/labels/<sheet>-<name>.csv`) are committed with `git add -f` only
once both have finished. Then: `just label-metrics agreement` per sheet, disagreements discussed,
and `just label-metrics confusion` against the key.
