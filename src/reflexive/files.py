"""Small file helpers shared by the data scripts: JSON Lines, checksums, paths to print."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]  # the repository (the package is installed editable)


class BadJsonLine(ValueError):
    """A line of a ``.jsonl`` file isn't valid JSON, e.g. a run stopped halfway through writing it."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Every record in a JSON Lines file, skipping blank lines.

    Raises ``BadJsonLine`` naming the file and line, rather than a bare ``JSONDecodeError``.
    """
    out = []
    for n, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise BadJsonLine(
                f"{shown(path)}, line {n}: not valid JSON ({e.msg}); "
                "a half-written last line? Remove it and re-run."
            ) from None
    return out


def sha256(path: Path) -> str:
    """The file's SHA-256, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: Path) -> str:
    """A path to print: relative to the repository when it's inside it, else absolute."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
