"""Tools the agent can call. Starter task: read-only SQL over the BioGRID database.

Swap or extend this module for your own environment. Each tool is a plain Python
function with a Google-style docstring — AgentTune turns the signature and docstring
into the tool schema the model sees, so keep them accurate.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

_DB_TOOL = None


def setup_biogrid(db_path: str = "data/biogrid.db") -> None:
    """Build the SQLite DB from the HF dataset on first use (idempotent)."""
    global _DB_TOOL
    from agenttune.agentic.tools.builtin.sql import SQLDatabaseTool

    path = Path(db_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists()
    _DB_TOOL = SQLDatabaseTool(f"sqlite:///{path}")
    if fresh:
        _DB_TOOL.create_from_dataset(dataset_name="qgallouedec/biogrid",
                                     table_name="interactions", split="train")


def query_biogrid(sql_command: str) -> Any:
    """
    Execute a read-only SQL query on the BioGRID database (table: interactions).

    Args:
        sql_command: The SQL query to execute.

    Returns:
        A list of rows, or an error message.
    """
    if _DB_TOOL is None:
        setup_biogrid()
    result = _DB_TOOL.execute(action="sql_db_query", input=sql_command)
    return result.output if result.success else {"error": result.error}


def get_tools(names: list[str]) -> list[Callable]:
    available = {"query_biogrid": query_biogrid}
    missing = [n for n in names if n not in available]
    if missing:
        raise KeyError(f"unknown tool(s) {missing}; available: {sorted(available)}")
    return [available[n] for n in names]
