"""Customer Lookup: a toy tool-use task for the GRPO warm-up (SEL-34, docs/experiments/GRPO_WARMUP.md).

A fake customer database (fixed seed) behind two tools. Each question asks for one field of
one customer; the values are random, so the model can't guess them and has to call the tools.

- **L1** (one call): "What is the balance of customer C1023?" → ``get_customer``
- **L2** (two calls): "How much data does Priya Shah have left?" → ``find_customer``,
  then ``get_customer``

Questions are split **by customer**: eval customers never appear in training, so the eval
measures tool use, not memorised answers. A hallucinated ``customer_id`` (one that doesn't
exist) is the toy version of τ²-bench's Arg hallucination.

Stdlib only (no torch, no AgentTune), so it's testable on a laptop. The rollout conversations
it scores come from AgentTune's rollout (``conversations`` / ``responses`` reward kwargs).
"""

from __future__ import annotations

import json
import random
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

SEED = 42
N_CUSTOMERS = 200
N_EVAL_CUSTOMERS = 40

PLANS = ("Basic", "Plus", "Premium", "Unlimited", "Student", "Family")
FIRST = ("Priya", "Tom", "Aisha", "Liam", "Mei", "Omar", "Sofia", "Jack", "Zara", "Noah",
         "Elena", "Ravi", "Chloe", "Kwame", "Hana", "Lucas", "Fatima", "Ben", "Ines", "Yusuf")
LAST = ("Shah", "Brown", "Khan", "Murphy", "Chen", "Haddad", "Rossi", "Taylor", "Ali", "Smith",
        "Cho", "Patel", "Dubois", "Mensah", "Sato", "Silva", "Begum", "Walsh", "Costa", "Okafor")

# field -> (how the question names it, tolerance for a numeric answer; None = text)
FIELDS = {
    "balance": ("the account balance", 0.005),
    "data_left_gb": ("the mobile data left, in GB,", 0.05),
    "plan": ("the plan name", None),
}

PREAMBLE = (
    "You are a customer-support assistant. Customer records are only available through the "
    "tools. Never guess a value: look it up. When you have the value, reply with just the "
    "answer wrapped in stars, for example *42.10*."
)

# How the tools and their results are shown to a model whose chat template can't render tools
# (Tiny Aya). AgentTune's defaults say "use this result to answer" after every tool result,
# which stops the model before a second call, so L2 questions could never be answered (first
# baseline, 5 Nov: L2 correct 0%, exactly one call per rollout). Passed to both training
# (train_grpo.py) and eval (eval_lookup.py), so both see the same harness.
TOOLS_FALLBACK_PROMPT = (
    "You can call these tools:\n{tools}\n\n"
    "To call tools, reply with only a JSON list, one object per call, and no other text: "
    '[{{"tool_name": "<name>", "parameters": {{"<argument>": <value>}}}}]. '
    "Use a list even for one call. "
    'A user turn that starts with "Tool result" is the output of your call, not a new '
    "message from the user. You can make several calls in a row, using one call's result "
    "in the next. When you have everything you need, answer the user's original question."
)
TOOL_RESULT_FORMAT = (
    "Tool result for {name}: {content}\n"
    "If you need more information, call another tool. Otherwise, answer the original question."
)
HARNESS = {"tools_fallback_prompt": TOOLS_FALLBACK_PROMPT, "tool_result_format": TOOL_RESULT_FORMAT}


# --- The database and its tools ---------------------------------------------------------------


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


def find_customer(name: str) -> dict:
    """Find a customer's ID from their full name.

    Args:
        name: The customer's full name, e.g. "Priya Shah".

    Returns:
        A dict with the customer_id, or an error if no customer has that name.
    """
    wanted = " ".join(str(name).split()).lower()
    for c in customers():
        if c["name"].lower() == wanted:
            return {"customer_id": c["customer_id"]}
    return {"error": f"No customer named {name!r}"}


def get_customer(customer_id: str) -> dict:
    """Get a customer's record: name, plan, balance and mobile data left.

    Args:
        customer_id: The customer's ID, e.g. "C1023".

    Returns:
        The customer record, or an error if the ID doesn't exist.
    """
    c = _by_id().get(str(customer_id).strip().upper())
    return dict(c) if c else {"error": f"No customer with ID {customer_id!r}"}


TOOLS = [find_customer, get_customer]


# --- Questions and splits ---------------------------------------------------------------------


def eval_customer_ids() -> set[str]:
    """The 40 customers held out for evaluation (fixed by SEED)."""
    ids = [c["customer_id"] for c in customers()]
    return set(random.Random(SEED + 1).sample(ids, N_EVAL_CUSTOMERS))


def questions(split: str, l2_share: float = 0.5, per_customer: int = 2) -> list[dict]:
    """Question rows for ``split`` ("train" or "eval"); the same arguments give the same rows.

    Each row: ``qid``, ``customer_id``, ``level`` (1 or 2), ``field``, ``question``, ``answer``.
    ``l2_share`` is the share of two-call (name) questions: raise or lower it to move the
    base model's success rate into the 20-70% band where GRPO learns.
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split must be 'train' or 'eval', not {split!r}")
    held_out = eval_customer_ids()
    rng = random.Random(f"{SEED}-{split}-{l2_share}-{per_customer}")
    rows = []
    for c in customers():
        if (c["customer_id"] in held_out) != (split == "eval"):
            continue
        for field in rng.sample(list(FIELDS), per_customer):
            level = 2 if rng.random() < l2_share else 1
            who = c["name"] if level == 2 else f"customer {c['customer_id']}"
            rows.append({
                "qid": f"{split}-{len(rows):04d}",
                "customer_id": c["customer_id"],
                "level": level,
                "field": field,
                "question": f"What is {FIELDS[field][0]} for {who}?",
                "answer": c[field],
            })
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


def starred(text: str) -> str | None:
    """The last ``*...*`` answer in ``text`` (ignoring ``**bold**``), or None."""
    found = re.findall(r"(?<!\*)\*([^*\n]{1,80})\*(?!\*)", text or "")
    return found[-1].strip() if found else None


def _matches(predicted: str | None, expected: str, field: str) -> bool:
    if predicted is None:
        return False
    tol = FIELDS[field][1]
    if tol is None:
        return predicted.strip(" .").lower() == expected.lower()
    num = re.search(r"-?\d+(?:\.\d+)?", predicted.replace(",", ""))
    return num is not None and abs(float(num.group()) - float(expected)) <= tol


def score(conversation: list[dict], final_text: str, answer: str, field: str) -> dict:
    """Score one rollout. The components (each 0 or 1, plus counts):

    - ``correct``: the starred answer matches the true value
    - ``tool_ok``: at least one tool call, and every call named a real tool, used a real
      customer ID and got a result (no errors)
    - ``format``: exactly one starred answer in the final reply
    - ``n_tool_calls``, ``hallucinated_ids`` (``get_customer`` calls with an ID that doesn't exist)
    """
    calls = tool_calls(conversation)
    names = {f.__name__ for f in TOOLS}
    ids = _by_id()
    hallucinated = sum(
        1 for name, args in calls
        if name == "get_customer" and str(args.get("customer_id", "")).strip().upper() not in ids
    )
    # Tool results are "tool" turns, or "user" turns starting "Tool result" when AgentTune folds
    # them for a model without tool support (Tiny Aya).
    errors = sum(
        1 for m in conversation or []
        if isinstance(m, dict) and "'error'" in str(m.get("content", ""))
        and (m.get("role") == "tool"
             or (m.get("role") == "user" and str(m.get("content", "")).startswith("Tool result")))
    )
    tool_ok = bool(calls) and hallucinated == 0 and errors == 0 and all(n in names for n, _ in calls)
    predicted = starred(final_text)
    n_starred = len(re.findall(r"(?<!\*)\*([^*\n]{1,80})\*(?!\*)", final_text or ""))
    return {
        "correct": float(_matches(predicted, answer, field)),
        "tool_ok": float(tool_ok),
        "format": float(n_starred == 1),
        "n_tool_calls": len(calls),
        "hallucinated_ids": hallucinated,
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


def make_reward_fn(weights: dict | None = None, log_path: str | Path | None = None):
    """A GRPO reward function: the weighted sum of :func:`score`'s components.

    TRL passes the dataset columns (``answer``, ``field``, ``qid``, ``level``) and AgentTune's
    rollout fields (``conversations``, ``responses``) as keyword arguments. Every scored
    rollout is appended to ``log_path`` as one JSON line (open it with
    ``pd.read_json(path, lines=True)``): AgentTune's own trajectories.jsonl leaves the reward
    empty. The first call also writes ``reward_inputs_debug.json`` next to it, showing exactly
    what the reward function received.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    log_path = Path(log_path) if log_path else None
    lock = threading.Lock()
    state = {"calls": 0}

    def lookup_reward(completions, answer=None, field=None, qid=None, level=None,
                      conversations=None, responses=None, **kwargs) -> list[float]:
        n = len(completions)
        convs = conversations if conversations is not None else [None] * n
        state["calls"] += 1
        if log_path and state["calls"] == 1:
            _write_debug(log_path.parent / "reward_inputs_debug.json", completions, convs, kwargs)
        trainer_state = kwargs.get("trainer_state")
        step = getattr(trainer_state, "global_step", None)
        rewards, rows = [], []
        for i in range(n):
            text = (responses[i] if responses is not None and responses[i] else
                    _final_text(convs[i], completions[i]))
            s = score(convs[i] or [], text, answer[i], field[i])
            r = sum(w[k] * s[k] for k in w)
            rewards.append(r)
            rows.append({"step": step, "call": state["calls"], "qid": qid[i] if qid else None,
                         "level": level[i] if level else None, "field": field[i],
                         "expected": answer[i], **s, "reward": r, "final_text": text[:300]})
        if log_path:
            with lock, open(log_path, "a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
        return rewards

    return lookup_reward


def _write_debug(path: Path, completions, conversations, kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    debug = {
        "kwarg_names": sorted(kwargs) + ["completions", "conversations"],
        "completion_type": type(completions[0]).__name__ if completions else None,
        "completion_0": completions[0] if completions else None,
        "conversation_0": conversations[0] if conversations else None,
    }
    path.write_text(json.dumps(debug, indent=2, default=str))
