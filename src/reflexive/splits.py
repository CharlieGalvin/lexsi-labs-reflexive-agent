"""The frozen τ²-bench task splits (SEL-36): train / dev / test task IDs per domain.

``configs/splits.json`` is the single source of truth. It's made once by
``scripts/data/make_splits.py`` from τ²-bench v2.0.0's own ``split_tasks.json``:

- **test** is τ²-bench's official test split, unchanged. It's only for the final evaluation.
  It's never used for healing, teacher fixes, DSPy or choosing anything.
- **dev** is a fixed random sample of τ²-bench's *train* tasks (15 retail, 6 airline,
  15 telecom), split **by task**. Its trajectories are never trained on.
- **train** is the rest of τ²-bench's train split: what the student trains on.

Stdlib only, so it works without τ²-bench installed.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

SPLITS_PATH = Path(__file__).resolve().parents[2] / "configs" / "splits.json"
DOMAINS = ("retail", "airline", "telecom")
SPLITS = ("train", "dev", "test")


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict:
    """The whole splits file (the metadata plus ``domains``)."""
    return json.loads(Path(path or SPLITS_PATH).read_text())


def task_ids(domain: str, split: str) -> list[str]:
    """The task IDs of one domain's split, in the file's order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    return list(load()["domains"][domain][split])


def split_of(domain: str, task_id: str) -> str:
    """Which split a task belongs to: 'train', 'dev' or 'test'. KeyError if it's in none."""
    d = load()["domains"][domain]
    for s in SPLITS:
        if str(task_id) in d[s]:
            return s
    raise KeyError(f"{domain} task {task_id!r} is in no split")


def assert_not_test(domain: str, task_id: str) -> None:
    """Raise if a task is in the test split. Call it wherever data could be trained on."""
    if split_of(domain, task_id) == "test":
        raise ValueError(f"{domain} task {task_id!r} is a TEST task: never heal, train or tune on it")
