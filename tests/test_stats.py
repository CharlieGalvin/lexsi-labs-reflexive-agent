"""reflexive.stats: the confidence ranges and paired tests behind the bake-off and arm comparisons."""

from __future__ import annotations

from reflexive.stats import binom_two_sided, holm, mcnemar, paired_bootstrap, wilson


def test_exact_binomial_matches_known_values() -> None:
    assert binom_two_sided(0, 3) == 0.25  # 3 vs 0 discordant tasks
    assert abs(binom_two_sided(1, 4) - 0.625) < 1e-9
    assert binom_two_sided(1, 2) == 1.0 and binom_two_sided(0, 0) == 1.0


def test_mcnemar_counts_only_discordant_tasks() -> None:
    a = {"t1": True, "t2": True, "t3": False, "t4": True}
    b = {"t1": True, "t2": False, "t3": False, "t4": False}
    assert mcnemar(a, b) == (2, 0, 0.5)


def test_wilson_and_holm() -> None:
    lo, hi = wilson(30, 30)
    assert round(lo, 2) == 0.89 and hi == 1.0
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_arm_a_vs_arm_c_on_telecom_train() -> None:
    """The numbers in docs/EXPERIMENTS.md: 5/74 vs 2/74, no shared passes, McNemar p = 0.45."""
    lo, hi = wilson(5, 74)
    assert (round(lo, 3), round(hi, 3)) == (0.029, 0.149)
    a = {t: t < 5 for t in range(74)}
    c = {t: 5 <= t < 7 for t in range(74)}
    assert mcnemar(a, c) == (5, 2, 0.453125)


def test_paired_bootstrap_is_reproducible_and_brackets_the_difference() -> None:
    a = {t: t % 2 == 0 for t in range(30)}  # 15/30
    b = {t: t % 3 == 0 for t in range(30)}  # 10/30
    lo, hi = paired_bootstrap(a, b, reps=2000)
    assert lo < 5 / 30 < hi and (lo, hi) == paired_bootstrap(a, b, reps=2000)


def test_wilson_matches_the_table_we_planned_the_bakeoff_with() -> None:
    lo, hi = wilson(0, 30)
    assert lo == 0 and round(hi, 3) == 0.114
    lo, hi = wilson(3, 30)
    assert (round(lo, 3), round(hi, 3)) == (0.035, 0.256)
