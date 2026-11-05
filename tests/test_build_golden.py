"""scripts/data/build_golden.py: the build stops (writes nothing) when a check fails."""

from __future__ import annotations

import pytest
from _scripts import load_script
from test_golden import SP, TOOLS, record

from reflexive import golden

bg = load_script("data/build_golden")
SYSTEM = {"telecom": (SP, TOOLS)}


def test_the_agents_view_check_passes_when_the_rebuilt_view_matches() -> None:
    r = record()
    saved = {**r, "agent_view": golden.agent_view(r["messages"], SP)}
    assert bg.check_views([saved, record()], SYSTEM) == 1  # the record without a saved view is skipped


def test_a_different_view_stops_the_build() -> None:
    r = record()
    saved = {**r, "agent_view": golden.agent_view(r["messages"], SP)[:-1]}
    with pytest.raises(bg.BuildError, match="agent's view differs from the saved one: telecom t1"):
        bg.check_views([saved], SYSTEM)


def test_no_saved_view_at_all_stops_the_build() -> None:
    with pytest.raises(bg.BuildError, match="no saved agent's view"):
        bg.check_views([record()], SYSTEM)


def test_the_totals_are_counted_not_typed(capsys: pytest.CaptureFixture[str]) -> None:
    doms = {d: {"train": {"trajectories": 1, "tasks": 2}, "dev": {"trajectories": 0, "tasks": 1}}
            for d in ("retail", "airline", "telecom")}
    bg.print_table({"domains": doms})
    assert "total       3 of   6        0 of  3" in capsys.readouterr().out
