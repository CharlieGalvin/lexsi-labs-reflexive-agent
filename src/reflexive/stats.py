"""Statistics for comparing pass rates: confidence ranges and paired tests. Standard library only.

Used by the bake-off and phase 2 reports, the bake-off run, and anything that compares two arms
on the same tasks. "Paired" functions take ``{task: passed}`` dicts and use only the shared
tasks. Any task key works; ``paired_bootstrap`` sorts them, so they must be comparable.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence range for a pass rate of k out of n (Wilson score interval)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test of k successes in n trials at p = 0.5."""
    if n == 0:
        return 1.0
    pk = math.comb(n, k) / 2**n
    return min(
        1.0, sum(math.comb(n, i) / 2**n for i in range(n + 1) if math.comb(n, i) / 2**n <= pk * (1 + 1e-9))
    )


def mcnemar[K](a: Mapping[K, bool], b: Mapping[K, bool]) -> tuple[int, int, float]:
    """Paired exact McNemar test on shared tasks: (only a passed, only b passed, p-value)."""
    shared = a.keys() & b.keys()
    only_a = sum(a[t] and not b[t] for t in shared)
    only_b = sum(b[t] and not a[t] for t in shared)
    return only_a, only_b, binom_two_sided(min(only_a, only_b), only_a + only_b)


def holm(pvals: list[float]) -> list[float]:
    """Holm-adjusted p-values, for several comparisons at once (same order as given)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_bootstrap[K: SupportsRichComparison](
    a: Mapping[K, bool], b: Mapping[K, bool], reps: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% bootstrap interval for pass-rate(a) - pass-rate(b), resampling shared tasks."""
    shared = sorted(a.keys() & b.keys())
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        s = [rng.choice(shared) for _ in shared]
        diffs.append(sum(a[t] - b[t] for t in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]


def cluster_bootstrap[K](
    clusters: Mapping[K, list[bool]], reps: int = 10_000, seed: int = 0
) -> tuple[float, float] | None:
    """95% percentile-bootstrap range of a pass rate, resampling whole clusters (a task's samples).

    Two samples of one task aren't independent, so resampling them separately would make the
    range too narrow. None when there is nothing to resample."""
    groups = [g for g in clusters.values() if g]
    if not groups:
        return None
    rng = random.Random(seed)
    rates = []
    for _ in range(reps):
        pick = rng.choices(groups, k=len(groups))
        rates.append(sum(map(sum, pick)) / sum(map(len, pick)))
    rates.sort()
    return rates[int(0.025 * reps)], rates[min(int(0.975 * reps), reps - 1)]


def mcnemar_power(p_a: float, p_b: float, n: int, sims: int = 4000, seed: int = 1) -> float:
    """Power of the exact McNemar test at alpha 0.05, by simulation (independent tasks)."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(sims):
        oa = ob = 0
        for _ in range(n):
            x, y = rng.random() < p_a, rng.random() < p_b
            oa += x and not y
            ob += y and not x
        hits += binom_two_sided(min(oa, ob), oa + ob) < 0.05
    return hits / sims
