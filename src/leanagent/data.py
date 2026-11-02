"""Dataset loading. Every row ends up as {"id", "prompt": [chat turns], "answer"}.

The starter task is BioGRID yes/no QA (the example AgentTune ships with). To make the
hallucination half of the thesis measurable we mix in questions the database cannot
answer (gold = "UNANSWERABLE"): the right behaviour is to check, then say so.
"""

from __future__ import annotations

import random
import textwrap
from typing import Any

from .trajectory import UNANSWERABLE

PREAMBLE = textwrap.dedent("""\
    You have access to the BioGRID SQLite database (table: interactions).
    Use as few SQL queries as you need, and never repeat an identical query.
    Answer only from query results. If the database cannot answer the question, say
    "The information is not available in the database."
    Put the final answer in stars, e.g. *Yes* or *No*.
    """)

# Questions about attributes the interactions table does not hold. TODO(team): check
# these against the real schema and grow the list (or generate them with CuratorKIT).
UNANSWERABLE_TEMPLATES = [
    "In what year was the gene {g} first discovered?",
    "Which laboratory first sequenced the gene {g}?",
    "What is the market price of an assay kit for the gene {g}?",
    "Which Nobel laureate studied the gene {g}?",
]
EXAMPLE_GENES = ["TP53", "BRCA1", "MYC", "EGFR", "CDC28", "RAD51", "ACT1", "GAL4"]


def _row(i: int, question: str, answer: str, preamble: str) -> dict[str, Any]:
    return {"id": f"q{i:05d}",
            "prompt": [{"role": "user", "content": f"{preamble}\nQuestion: {question}"}],
            "answer": answer}


def load_task(cfg: dict[str, Any], split: str):
    """Returns a ``datasets.Dataset`` for ``split`` ("train" or "eval")."""
    from datasets import Dataset, load_dataset

    seed = cfg.get("seed", 42)
    raw = load_dataset(cfg["hf_dataset"], split="train")
    prefix = cfg.get("question_prefix")
    if prefix:
        raw = raw.filter(lambda ex: ex["question"].startswith(prefix))
    raw = raw.shuffle(seed=seed)

    n_eval = cfg.get("n_eval", 200)
    part = raw.select(range(n_eval)) if split == "eval" else raw.select(range(n_eval, len(raw)))
    if split == "train" and cfg.get("n_train"):
        part = part.select(range(min(cfg["n_train"], len(part))))

    rows = [_row(i, ex["question"], ex["answer"], PREAMBLE) for i, ex in enumerate(part)]

    frac = cfg.get("unanswerable_fraction", 0.0)
    rng = random.Random(seed + (0 if split == "train" else 1))
    for _ in range(int(len(rows) * frac)):
        q = rng.choice(UNANSWERABLE_TEMPLATES).format(g=rng.choice(EXAMPLE_GENES))
        rows.append(_row(len(rows), q, UNANSWERABLE, PREAMBLE))
    rng.shuffle(rows)
    return Dataset.from_list(rows)
