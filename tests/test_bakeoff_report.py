"""scripts/teacher/bakeoff_report.py: the projected cost (the statistics are in test_stats.py)."""

from __future__ import annotations

from _scripts import load_script

br = load_script("teacher/bakeoff_report")


def test_projection_stops_at_the_first_pass():
    perfect = br.project(1.0, 0.01)
    assert perfect["attempts_per_task"] == 1.0
    assert round(perfect["fixes_cycle1"]) == round(178 * 0.97)
    assert br.project(0.5, 0.01)["total"] > perfect["total"]
