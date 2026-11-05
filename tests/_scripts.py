"""Load a script from scripts/ as a module, for tests of its command-line logic.

Scripts aren't a package (each runs as ``python scripts/<folder>/<name>.py``), so tests load
them by path. Shared logic lives in ``src/reflexive`` and is imported normally.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(rel: str) -> ModuleType:
    """``load_script("data/publish_golden")`` → the module of ``scripts/data/publish_golden.py``."""
    path = SCRIPTS / f"{rel}.py"
    name = "script_" + rel.replace("/", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses and pickling look the module up by name
    spec.loader.exec_module(mod)
    return mod
