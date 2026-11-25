"""The "doesn't match" rule: a change that differs from what the customer asked for and agreed to,
or from what the agent's own lookups showed (wrong item, wrong order, more or fewer items, a flight
dropped). Written by hand on Sat 21 Nov, after meta-healing (SEL-91).

Where it came from: meta-healing's round 2 proposed a retail rule about item changes that passed the
numbers and failed review for being too broad, and its misfires flagged real errors of this kind
(airline 33: a flight change that dropped the flight the customer was keeping). The 12 Nov sort of
v2's failures had the same signal (about 15 unrequested changes and 14 wrong arguments of 55). So a
person wrote the tight version: the tools are listed, no "or similar", and the checker sees the
tool results in full, since a mismatch is only visible against them.

It runs through the same machinery as meta-healing's rules (`reflexive.meta_heal`), with the
pinned full ``gpt-4.1-2025-04-14`` (mini read meta-healing's rules too loosely), at temperature 0,
cached. The quoted evidence must be in the conversation, and a checker error never blocks. It reads
the conversation, the call and the policy only. With ``REFLEXIVE_INTENT_JUDGE=off`` it never fires.
"""

from __future__ import annotations

from reflexive.meta_heal import Candidate, as_rule

MODEL = "gpt-4.1-2025-04-14"  # full GPT-4.1, not mini: comparing IDs and options across long lookups is hard

TOOLS = (
    # retail: the changes that act on items, or on a whole order
    "modify_pending_order_items",
    "exchange_delivered_order_items",
    "return_delivered_order_items",
    "cancel_pending_order",
    # airline: the changes to a reservation's flights, bags, or the whole reservation
    "update_reservation_flights",
    "update_reservation_baggages",
    "cancel_reservation",
)

WHAT = (
    "The change doesn't match what the customer asked for and agreed to, or what the agent's lookups showed."
)
QUESTION = (
    "Compare the PROPOSED CHANGE with the conversation and the tool results. Is any part of it "
    "different from what the customer asked for and agreed to, or inconsistent with the agent's own "
    "lookups? Check: the order or reservation it acts on; each item ID and each new item ID (does each "
    "new item have exactly the options the customer asked for, according to the product lookups?); "
    "the number of items (more or fewer than the customer asked for, e.g. a whole order cancelled when "
    "they asked about some items); for flight changes, that every flight the customer is keeping is "
    "still in the list, with the dates and cabin they agreed; for bags, the numbers they asked for. "
    "Answer YES only for a concrete difference you can point to. Answer NO if the change matches, or "
    "if the customer never stated that detail."
)
BLOCK = (
    "this change may not match what the customer asked for or what your lookups showed. Check the "
    "order, the items and the numbers against the conversation, then describe the change and ask "
    "them to confirm."
)

RULE = Candidate(
    name="mismatch",
    what=WHAT,
    applies_to="call",
    question=QUESTION,
    live=True,
    block_message=BLOCK,
    tools=TOOLS,
    tool_chars=6000,
    model=MODEL,
)

mismatch = as_rule(RULE, kind="INTENT", prefix="")
