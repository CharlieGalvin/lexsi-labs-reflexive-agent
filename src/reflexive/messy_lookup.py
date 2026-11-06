"""Messy-ID Lookup: a one-tool-call task for a full GRPO run (docs/experiments/MESSY_LOOKUP.md).

The Customer Lookup warm-up (``toy_lookup``) showed Tiny Aya can make one tool call but never
chains two. Here every question needs exactly **one** ``get_customer`` call; the skill is
getting the **argument** right, the toy version of τ²-bench's Arg hallucination. Same database,
same scoring and reward (``toy_lookup.score`` / ``make_reward_fn``), same split by customer.

The question types (``level`` column), easiest first:

- ``clean``: "customer C1023"
- ``lower_dash``: "c-1023"
- ``spaced``: "C 1023"
- ``number_only``: "account number 1023" (the model must add the C)
- ``spelled``: "C one zero two three"
- ``distractor``: two IDs in one message, only one of them is the customer asked about

``get_customer`` only accepts the standard form (case aside), so a badly copied ID returns an
error; ``hallucinated_ids`` counts these, ``wrong_customer`` counts the distractor mistakes.
Stdlib only.
"""

from __future__ import annotations

import random

from reflexive import toy_lookup
from reflexive.toy_lookup import FIELDS, SEED, customers, eval_customer_ids

STYLES = ("clean", "lower_dash", "spaced", "number_only", "spelled", "distractor")
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def get_customer(customer_id: str) -> dict:
    """Get a customer's record: name, plan, balance and mobile data left.

    Args:
        customer_id: The customer's ID in its standard form: the capital letter C followed by
            four digits, with no spaces or dashes. Customers sometimes write their ID
            differently; convert it to the standard form first. Never make one up.

    Returns:
        The customer record, or an error if the ID doesn't exist.
    """
    return toy_lookup.get_customer(customer_id)


TOOLS = [get_customer]


def _write_id(style: str, cid: str) -> str:
    """How a question writes the ID ``cid`` (e.g. "C1023") in ``style``."""
    digits = cid[1:]
    return {
        "clean": f"customer {cid}",
        "lower_dash": f"customer c-{digits}",
        "spaced": f"customer C {digits}",
        "number_only": f"account number {digits}",
        "spelled": "customer C " + " ".join(DIGITS[int(d)] for d in digits),
    }[style]


def _question(style: str, field: str, cid: str, other_cid: str, rng: random.Random) -> str:
    what = FIELDS[field][0]
    if style != "distractor":
        return f"What is {what} for {_write_id(style, cid)}?"
    # Two real IDs; the question is about one of them, in either position.
    if rng.random() < 0.5:
        return f"Hi, I'm customer {other_cid}. What is {what} for my partner, customer {cid}?"
    return f"My partner is customer {other_cid}, but I'm asking about me: customer {cid}. What is {what}?"


def questions(split: str, styles: dict[str, float] | None = None, per_customer: int = 3) -> list[dict]:
    """Question rows for ``split`` ("train" or "eval"); the same arguments give the same rows.

    Each row: ``qid``, ``customer_id``, ``level`` (the question type), ``field``,
    ``question``, ``answer``. ``styles`` weights the question types (default: equal); move the
    weight towards the hard types if the base model is too good, and the other way if not.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', not {split!r}")
    weights = styles or {s: 1.0 for s in STYLES}
    unknown = set(weights) - set(STYLES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}; choose from {STYLES}")
    names, w = zip(*weights.items(), strict=True)
    held_out = eval_customer_ids()
    pool = [c for c in customers() if (c["customer_id"] in held_out) == (split == "eval")]
    rng = random.Random(f"messy-{SEED}-{split}-{sorted(weights.items())}-{per_customer}")
    rows = []
    for c in pool:
        for field in rng.sample(list(FIELDS), per_customer):
            style = rng.choices(names, weights=w)[0]
            other = rng.choice([o for o in pool if o is not c])  # same split: no leakage
            rows.append({
                "qid": f"{split}-{len(rows):04d}",
                "customer_id": c["customer_id"],
                "level": style,
                "field": field,
                "question": _question(style, field, c["customer_id"], other["customer_id"], rng),
                "answer": c[field],
            })
    return rows


prompt = toy_lookup.prompt
make_reward_fn = toy_lookup.make_reward_fn
score = toy_lookup.score
HARNESS = toy_lookup.HARNESS
