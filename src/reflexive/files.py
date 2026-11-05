"""Small file helpers shared by the data scripts: JSON Lines, checksums, paths to print."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """The repository: this package's own checkout when it's installed editable (the usual setup),
    else the nearest folder at or above the current directory that has ``configs/splits.json``."""
    marker = Path("configs") / "splits.json"
    here = Path(__file__).resolve().parents[2]
    if (here / marker).exists():
        return here
    cwd = Path.cwd().resolve()
    return next((d for d in (cwd, *cwd.parents) if (d / marker).exists()), cwd)


ROOT = _repo_root()


class BadJsonLine(ValueError):
    """A line of a ``.jsonl`` file isn't valid JSON, e.g. a run stopped halfway through writing it."""


def read_jsonl(path: Path, *, partial_last_ok: bool = False) -> list[dict[str, Any]]:
    """Every record in a JSON Lines file, skipping blank lines.

    Raises ``BadJsonLine`` naming the file and line, rather than a bare ``JSONDecodeError``.
    ``partial_last_ok``: a bad *last* line (a run stopped while writing it) is skipped with a
    note on stderr instead; a bad line anywhere else still raises.
    """
    lines = [(n, line) for n, line in enumerate(path.read_text().splitlines(), start=1) if line.strip()]
    out = []
    for i, (n, line) in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if partial_last_ok and i == len(lines) - 1:
                print(f"note: {shown(path)}, line {n} is half-written (skipped)", file=sys.stderr)
                break
            raise BadJsonLine(
                f"{shown(path)}, line {n}: not valid JSON ({e.msg}); "
                "a half-written last line? Remove it and re-run."
            ) from None
    return out


def resume_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records an interrupted run saved, ready to append to.

    Like ``read_jsonl(path, partial_last_ok=True)``, but a half-written last line is also cut
    from the file: otherwise the next record appended would be glued onto it and lost too.
    """
    recs = read_jsonl(path, partial_last_ok=True)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(recs) < len(lines):  # cut the half-written last line; keep the others byte for byte
        path.write_text("".join(line + "\n" for line in lines[:-1]))
    return recs


def sha256(path: Path) -> str:
    """The file's SHA-256, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shown(path: Path) -> str:
    """A path to print: relative to the repository when it's inside it, else absolute."""
    p = Path(path).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
