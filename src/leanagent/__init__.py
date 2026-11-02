"""leanagent: RL on tool-calling trajectories for fewer, better tool calls and fewer hallucinations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

__version__ = "0.1.0"


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config; ``overrides`` are dotted ``key=value`` strings, e.g. train.max_steps=10."""
    import yaml

    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg
