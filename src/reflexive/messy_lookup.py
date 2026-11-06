"""Messy-ID Lookup: a one-tool-call task for a full GRPO run (docs/experiments/MESSY_LOOKUP.md).

A fake customer database (fixed seed) behind one tool, ``get_customer``. Each question asks for
one field of one customer; the values are random, so the model can't guess them and has to call
the tool. Every question needs exactly **one** call; the skill is getting the **argument** right,
the toy version of τ²-bench's Arg hallucination. The question types (``level`` column):

- ``clean``: "customer C1023"
- ``lower_dash``: "c-1023"
- ``spaced``: "C 1023"
- ``number_only``: "account number 1023" (the model must add the C)
- ``spelled``: "C one zero two three"
- ``distractor``: two IDs in one message, only one of them is the customer asked about

``get_customer`` only accepts the standard form (case aside), so a badly copied ID returns an
error; ``hallucinated_ids`` counts these, ``wrong_customer`` counts real IDs of the wrong person.
Questions are split **by customer**: eval customers never appear in training.

Stdlib only (no torch, no AgentTune), so it's testable on a laptop. The rollout conversations
it scores come from AgentTune's rollout (``conversations`` / ``responses`` reward kwargs).
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

SEED = 42
N_CUSTOMERS = 200
N_EVAL_CUSTOMERS = 40

PLANS = ("Basic", "Plus", "Premium", "Unlimited", "Student", "Family")
FIRST = (
    "Priya",
    "Tom",
    "Aisha",
    "Liam",
    "Mei",
    "Omar",
    "Sofia",
    "Jack",
    "Zara",
    "Noah",
    "Elena",
    "Ravi",
    "Chloe",
    "Kwame",
    "Hana",
    "Lucas",
    "Fatima",
    "Ben",
    "Ines",
    "Yusuf",
)
LAST = (
    "Shah",
    "Brown",
    "Khan",
    "Murphy",
    "Chen",
    "Haddad",
    "Rossi",
    "Taylor",
    "Ali",
    "Smith",
    "Cho",
    "Patel",
    "Dubois",
    "Mensah",
    "Sato",
    "Silva",
    "Begum",
    "Walsh",
    "Costa",
    "Okafor",
)

# field -> (how the question names it, tolerance for a numeric answer; None = text)
FIELDS = {
    "balance": ("the account balance", 0.005),
    "data_left_gb": ("the mobile data left, in GB,", 0.05),
    "plan": ("the plan name", None),
}

STYLES = ("clean", "lower_dash", "spaced", "number_only", "spelled", "distractor")
DIGITS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")

PREAMBLE = (
    "You are a customer-support assistant. Customer records are only available through the "
    "tools. Never guess a value: look it up. When you have the value, reply with just the "
    "answer wrapped in stars, for example *42.10*."
)

# How the tool and its results are shown to a model whose chat template can't render tools
# (Tiny Aya). AgentTune's default result wording says "use this result to answer", even after
# an error; ours lets the model retry once with a corrected ID. Training (train_grpo.py) and
# eval (eval_lookup.py) both use it, so both see the same harness.
TOOLS_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user: use it to answer the user's original question."
)
TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\n"
    "If this is an error, you may call the tool again with a corrected argument. "
    "Otherwise, answer the original question."
)
HARNESS = {"tools_fallback_prompt": TOOLS_FALLBACK_PROMPT, "tool_result_format": TOOL_RESULT_FORMAT}


# --- The database and its tool ----------------------------------------------------------------


@lru_cache(maxsize=1)
def customers() -> tuple[dict, ...]:
    """The fixed database: 200 customers with unique names and random field values."""
    rng = random.Random(SEED)
    names = rng.sample([f"{f} {last}" for f in FIRST for last in LAST], N_CUSTOMERS)
    return tuple(
        {
            "customer_id": f"C{1000 + i}",
            "name": name,
            "plan": rng.choice(PLANS),
            "balance": f"{rng.uniform(0, 500):.2f}",
            "data_left_gb": f"{rng.uniform(0, 50):.1f}",
        }
        for i, name in enumerate(names)
    )


@lru_cache(maxsize=1)
def _by_id() -> dict[str, dict]:
    return {c["customer_id"]: c for c in customers()}


# The docstring is what the model sees. Keep real example IDs out of it: Tiny Aya copies them.
def get_customer(customer_id: str) -> dict:
    """Get a customer's record: name, plan, balance and mobile data left.

    Args:
        customer_id: The customer's ID in its standard form: the capital letter C followed by
            four digits, with no spaces or dashes. Customers sometimes write their ID
            differently; convert it to the standard form first. Never make one up.

    Returns:
        The customer record, or an error if the ID doesn't exist.
    """
    c = _by_id().get(str(customer_id).strip().upper())
    return dict(c) if c else {"error": f"No customer with ID {customer_id!r}"}


TOOLS = [get_customer]


# --- Questions and splits ---------------------------------------------------------------------


def eval_customer_ids() -> set[str]:
    """The 40 customers held out for evaluation (fixed by SEED)."""
    ids = [c["customer_id"] for c in customers()]
    return set(random.Random(SEED + 1).sample(ids, N_EVAL_CUSTOMERS))


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


def _canonical(styles: Mapping[str, float]) -> dict[str, float]:
    """The weights in ``STYLES`` order, whole numbers as ints, so the same mix always gives the same
    seed and the same type choices: ``clean: 1`` and ``clean: 1.0``, or any key order, are one mix.

    This is the form the recorded runs (6 Nov) were seeded with: the config's int weights, in
    ``STYLES`` order. Keep it, or their question sets change."""
    unknown = set(styles) - set(STYLES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}; choose from {STYLES}")
    out: dict[str, float] = {}
    for name in STYLES:
        if name in styles:
            v = float(styles[name])
            out[name] = int(v) if v.is_integer() else v
    return out


def questions(split: str, styles: Mapping[str, float] | None = None, per_customer: int = 3) -> list[dict]:
    """Question rows for ``split`` ("train" or "eval"); the same arguments give the same rows.

    Each row: ``qid``, ``customer_id``, ``level`` (the question type), ``field``,
    ``question``, ``answer``. ``styles`` weights the question types (default: equal); move the
    weight towards the hard types if the base model is too good, and the other way if not.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', not {split!r}")
    weights = _canonical(styles or {s: 1 for s in STYLES})
    names, w = zip(*weights.items(), strict=True)
    held_out = eval_customer_ids()
    pool = [c for c in customers() if (c["customer_id"] in held_out) == (split == "eval")]
    rng = random.Random(f"messy-{SEED}-{split}-{sorted(weights.items())}-{per_customer}")
    rows = []
    for c in pool:
        for field in rng.sample(list(FIELDS), per_customer):
            style = rng.choices(names, weights=w)[0]
            other = rng.choice([o for o in pool if o is not c])  # same split: no leakage
            rows.append(
                {
                    "qid": f"{split}-{len(rows):04d}",
                    "customer_id": c["customer_id"],
                    "level": style,
                    "field": field,
                    "question": _question(style, field, c["customer_id"], other["customer_id"], rng),
                    "answer": c[field],
                }
            )
    return rows


def prompt(row: dict) -> list[dict]:
    """The chat prompt for one question (AgentTune adds the tool descriptions itself)."""
    return [{"role": "user", "content": f"{PREAMBLE}\n\nQuestion: {row['question']}"}]


# --- Scoring ----------------------------------------------------------------------------------


def tool_calls(conversation: list[dict]) -> list[tuple[str, dict]]:
    """(name, arguments) for every tool call in a rollout conversation, whatever its shape:
    ``{"function": {"name", "arguments"}}`` or ``{"name", "arguments"|"parameters"}``,
    with arguments as a dict or a JSON string."""
    calls = []
    for msg in conversation or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            args = fn.get("arguments", fn.get("parameters", {}))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append((str(fn.get("name", "")), args if isinstance(args, dict) else {}))
    return calls


# One *answer* (not **bold**). The opening star is followed by the answer itself, so a
# multiplication sign ("2 * 3 = *6*") never opens one; whitespace before the closing star is
# allowed: Tiny Aya writes "*45.1\n*".
_STARRED = re.compile(r"(?<!\*)\*(?=[^\s*])([^*]{1,80})\*(?!\*)")
# A customer ID inside an answer ("*C1023: 45.10*"): not the value, so it's ignored when matching
_ID_IN_ANSWER = re.compile(r"\b[Cc][- ]?\d{4}\b")


def starred(text: str) -> str | None:
    """The last ``*...*`` answer in ``text`` (ignoring ``**bold**``), or None."""
    found = [f.strip() for f in _STARRED.findall(text or "") if f.strip()]
    return found[-1] if found else None


def _matches(predicted: str | None, expected: str, field: str) -> bool:
    if predicted is None:
        return False
    tol = FIELDS[field][1]
    if tol is None:
        return predicted.strip(" .").lower() == expected.lower()
    # The first number once customer IDs are dropped: an answer without an ID scores as before
    num = re.search(r"-?\d+(?:\.\d+)?", _ID_IN_ANSWER.sub(" ", predicted).replace(",", ""))
    return num is not None and abs(float(num.group()) - float(expected)) <= tol


def score(
    conversation: list[dict], final_text: str, answer: str, field: str, customer_id: str | None = None
) -> dict:
    """Score one rollout. The components (each 0 or 1, plus counts):

    - ``correct``: the starred answer matches the true value
    - ``tool_ok``: the rollout fetched the asked-about customer's record (``customer_id``), and
      every call named a real tool, used a real ID of the right customer and got a result
    - ``format``: exactly one starred answer in the final reply
    - counts: ``n_tool_calls``; ``hallucinated_ids``: calls with an ID that doesn't exist
      (badly normalised or invented); ``wrong_customer``: real IDs of a different customer
    """
    calls = tool_calls(conversation)
    names = {f.__name__ for f in TOOLS}
    ids = _by_id()
    fetched = [
        str(args.get("customer_id", "")).strip().upper() for name, args in calls if name == "get_customer"
    ]
    hallucinated = sum(1 for i in fetched if i not in ids)
    wrong = sum(1 for i in fetched if i in ids and customer_id and i != customer_id)
    # Tool results are "tool" turns, or "user" turns starting "Tool result" when AgentTune folds
    # them for a model without tool support (Tiny Aya).
    errors = sum(
        1
        for m in conversation or []
        if isinstance(m, dict)
        and "'error'" in str(m.get("content", ""))
        and (
            m.get("role") == "tool"
            or (m.get("role") == "user" and str(m.get("content", "")).startswith("Tool result"))
        )
    )
    right_record = bool(fetched) and (customer_id is None or customer_id in fetched)
    tool_ok = (
        right_record
        and hallucinated == 0
        and wrong == 0
        and errors == 0
        and all(n in names for n, _ in calls)
    )
    n_starred = sum(1 for f in _STARRED.findall(final_text or "") if f.strip())
    predicted = starred(final_text)
    return {
        "correct": float(_matches(predicted, answer, field)),
        "tool_ok": float(tool_ok),
        "format": float(n_starred == 1),
        "n_tool_calls": len(calls),
        "hallucinated_ids": hallucinated,
        "wrong_customer": wrong,
        "predicted": predicted,
    }


DEFAULT_WEIGHTS = {"correct": 1.0, "tool_ok": 0.5, "format": 0.2}


def _final_text(conversation: list[dict] | None, completion: Any) -> str:
    for msg in reversed(conversation or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
            return str(msg["content"])
    if isinstance(completion, list):  # a list of message dicts
        return _final_text(completion, None)
    return str(completion or "")


def make_reward_fn(
    weights: dict | None = None, log_path: str | Path | None = None
) -> Callable[..., list[float]]:
    """A GRPO reward function: the weighted sum of :func:`score`'s components.

    TRL passes the dataset columns (``answer``, ``field``, ``customer_id``, ``qid``, ``level``)
    and AgentTune's rollout fields (``conversations``, ``responses``) as keyword arguments.
    Every scored rollout is appended to ``log_path`` as one JSON line (open it with
    ``pd.read_json(path, lines=True)``): AgentTune's own trajectories.jsonl leaves the reward
    empty. The first call also writes ``reward_inputs_debug.json`` next to it, showing exactly
    what the reward function received.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    log_path = Path(log_path) if log_path else None
    if log_path and log_path.exists() and log_path.stat().st_size:
        # A run that stopped before run_summary.json left rows here: keep them apart, or the
        # re-run's curves would average both runs' step 1, 2, ...
        old = log_path.with_name(f"{log_path.stem}.stopped-{int(time.time())}{log_path.suffix}")
        log_path.rename(old)
    lock = threading.Lock()
    state = {"calls": 0}

    def lookup_reward(
        completions: list[Any],
        answer: list[str] | None = None,
        field: list[str] | None = None,
        qid: list[str] | None = None,
        level: list[str] | None = None,
        customer_id: list[str] | None = None,
        conversations: list[Any] | None = None,
        responses: list[str] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        if answer is None or field is None:
            raise ValueError("the training dataset needs 'answer' and 'field' columns")
        n = len(completions)
        convs = conversations if conversations is not None else [None] * n
        state["calls"] += 1
        if log_path and state["calls"] == 1:
            _write_debug(log_path.parent / "reward_inputs_debug.json", completions, convs, kwargs)
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        rewards, rows = [], []
        for i in range(n):
            text = (
                responses[i]
                if responses is not None and responses[i]
                else _final_text(convs[i], completions[i])
            )
            s = score(convs[i] or [], text, answer[i], field[i], customer_id[i] if customer_id else None)
            r = sum(w[k] * s[k] for k in w)
            rewards.append(r)
            rows.append(
                {
                    "step": step,
                    "call": state["calls"],
                    "qid": qid[i] if qid else None,
                    "level": level[i] if level else None,
                    "field": field[i],
                    "expected": answer[i],
                    **s,
                    "reward": r,
                    "final_text": text[:300],
                }
            )
        if log_path:
            with lock, open(log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        _log_parts_to_wandb(rows)
        return rewards

    return lookup_reward


_PARTS = ("correct", "tool_ok", "format", "n_tool_calls", "hallucinated_ids", "wrong_customer")


def _log_parts_to_wandb(rows: list[dict]) -> None:
    """Batch means of each reward part as ``reward_parts/<name>`` charts, when a W&B run is on.

    AgentTune folds every reward function into one, so TRL only logs the total. commit=False
    attaches these to the trainer's next log line, i.e. the same training step.
    """
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None or not rows:
        return
    wandb.log({f"reward_parts/{k}": sum(float(r[k]) for r in rows) / len(rows) for k in _PARTS}, commit=False)


def _write_debug(path: Path, completions: list[Any], conversations: list[Any], kwargs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    debug = {
        "kwarg_names": sorted(kwargs) + ["completions", "conversations"],
        "completion_type": type(completions[0]).__name__ if completions else None,
        "completion_0": completions[0] if completions else None,
        "conversation_0": conversations[0] if conversations else None,
    }
    path.write_text(json.dumps(debug, indent=2, default=str))
