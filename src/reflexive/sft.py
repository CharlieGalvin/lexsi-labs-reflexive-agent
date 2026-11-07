"""Tokenising SFT examples (SEL-65): learn the agent's turns, and nothing else.

An example from `just train-data` (``reagent.sft/v1``) is a whole conversation, exactly as Aya
reads it at run time, with ``train_on`` = the indices of the agent's turns. This turns it into
``input_ids`` and ``labels`` with the model's own chat template, so training sees the same tokens
vLLM builds when it serves the model (`scripts/pod/serve.sh`):

* every token of the conversation is input;
* **only the agent's turns are targets**: each turn's text and its end-of-turn token, not the
  turn header (``add_generation_prompt`` writes that at run time, so the model never has to
  produce it). The system prompt, the customer's turns (GPT-4.1 outputs) and the tool results
  are context, labelled ``-100``.

It relies on the template being **prefix-stable**: rendering the first ``i`` turns gives the
first tokens of rendering ``i + 1``. Aya's (Cohere's) template is; `encode` checks it on every
turn and refuses an example otherwise, rather than training on misaligned labels.

No torch here: a tokenizer is anything with Hugging Face's ``apply_chat_template``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

IGNORE = -100  # PyTorch's cross-entropy ignore index

Record = dict[str, Any]


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
    ) -> list[int]: ...


class NotPrefixStable(ValueError):
    """The chat template rewrote earlier turns when a turn was added: labels would be misaligned."""


def _ids(tok: ChatTokenizer, msgs: Sequence[dict[str, str]], gen: bool) -> list[int]:
    out = tok.apply_chat_template(list(msgs), tokenize=True, add_generation_prompt=gen)
    if isinstance(out, dict):  # newer transformers can return a BatchEncoding
        out = out["input_ids"]
    return list(out)


def encode(example: Record, tok: ChatTokenizer) -> dict[str, list[int]]:
    """``input_ids`` and ``labels`` for one conversation: targets on the agent's turns only."""
    msgs = [{"role": m["role"], "content": m["content"]} for m in example["messages"]]
    targets = set(example["train_on"])
    if any(msgs[i]["role"] != "assistant" for i in targets):
        raise ValueError(f"{example.get('id')}: train_on names a turn that isn't the agent's")
    ids: list[int] = []
    labels: list[int] = []
    for i in range(len(msgs)):
        if i in targets:
            head = _ids(tok, msgs[:i], gen=True)  # everything before the reply, header included
            if head[: len(ids)] != ids:
                raise NotPrefixStable(f"{example.get('id')}: turn {i} rewrote earlier tokens")
            labels += [IGNORE] * (len(head) - len(ids))
            ids = head
        full = _ids(tok, msgs[: i + 1], gen=False)
        if full[: len(ids)] != ids:
            raise NotPrefixStable(f"{example.get('id')}: turn {i} rewrote earlier tokens")
        new = full[len(ids) :]
        labels += new if i in targets else [IGNORE] * len(new)
        ids = full
    return {"input_ids": ids, "labels": labels}


def encode_all(
    examples: Sequence[Record], tok: ChatTokenizer, max_len: int
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    """Every example that fits in ``max_len`` tokens, and what was left out and why.

    A conversation is never cut: a cut one would end mid-task, and teach the model to stop there."""
    kept, too_long = [], []
    for e in examples:
        enc = encode(e, tok)
        if len(enc["input_ids"]) > max_len:
            too_long.append({"id": e.get("id"), "tokens": len(enc["input_ids"])})
            continue
        kept.append(enc)
    lengths = [len(k["input_ids"]) for k in kept]
    targets = [sum(t != IGNORE for t in k["labels"]) for k in kept]
    stats = {
        "examples": len(kept),
        "too_long": too_long,
        "tokens": {"max": max(lengths, default=0), "total": sum(lengths)},
        "target_tokens": sum(targets),
        "target_share": round(sum(targets) / max(sum(lengths), 1), 4),
    }
    return kept, stats


def pad_batch(batch: Sequence[dict[str, list[int]]], pad_id: int) -> dict[str, list[list[int]]]:
    """Right-pad a batch: pad tokens are masked out of attention and loss."""
    n = max(len(b["input_ids"]) for b in batch)
    return {
        "input_ids": [b["input_ids"] + [pad_id] * (n - len(b["input_ids"])) for b in batch],
        "attention_mask": [[1] * len(b["input_ids"]) + [0] * (n - len(b["input_ids"])) for b in batch],
        "labels": [b["labels"] + [IGNORE] * (n - len(b["labels"])) for b in batch],
    }
