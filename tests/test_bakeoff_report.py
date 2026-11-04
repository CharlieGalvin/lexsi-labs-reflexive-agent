"""scripts/bakeoff_report.py: the statistics behind the teacher bake-off write-up (SEL-58)."""

from __future__ import annotations

import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "bakeoff_report", pathlib.Path(__file__).parents[1] / "scripts" / "bakeoff_report.py")
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def test_exact_binomial_matches_known_values():
    assert br.binom_two_sided(0, 3) == 0.25          # 3 vs 0 discordant tasks
    assert abs(br.binom_two_sided(1, 4) - 0.625) < 1e-9
    assert br.binom_two_sided(1, 2) == 1.0 and br.binom_two_sided(0, 0) == 1.0


def test_mcnemar_counts_only_discordant_tasks():
    a = {"t1": True, "t2": True, "t3": False, "t4": True}
    b = {"t1": True, "t2": False, "t3": False, "t4": False}
    assert br.mcnemar(a, b) == (2, 0, 0.5)


def test_wilson_and_holm():
    lo, hi = br.wilson(30, 30)
    assert round(lo, 2) == 0.89 and hi == 1.0
    assert br.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_projection_stops_at_the_first_pass():
    perfect = br.project(1.0, 0.01)
    assert perfect["attempts_per_task"] == 1.0
    assert round(perfect["fixes_cycle1"]) == round(178 * 0.97)
    assert br.project(0.5, 0.01)["total"] > perfect["total"]
