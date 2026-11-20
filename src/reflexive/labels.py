"""Scoring a detector against a reference, and two labellers against each other (SEL-85, SEL-98).

Standard library only. Two questions, one module:

* **Confusion:** a detector's flags (the prediction) against a reference (the truth): the Tool/Arg/Loop
  classifier against the human labels, or the Stage 2 checker against τ²-bench's verdicts. Gives the
  counts (TP, FP, FN, TN) and precision, recall, F1, false-positive rate and accuracy, each with a
  95% interval.
* **Agreement:** two labellers' categories on the same items: raw agreement and Cohen's kappa, with
  a 95% interval, and the items they disagree on (to settle afterwards).

Intervals: Wilson for the proportions when items are independent. When items come in clusters (a
task's two samples), every interval is a percentile bootstrap that resamples whole clusters, as in
``reflexive.stats.cluster_bootstrap``: two samples of one task aren't independent.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass

from reflexive.stats import wilson

REPS = 10_000
TRUE_WORDS = {"1", "true", "t", "yes", "y"}
FALSE_WORDS = {"0", "false", "f", "no", "n", ""}


def as_bool(value: object, positive: str | None = None) -> bool:
    """A label as a bool: ``positive`` if given (exact match, case-insensitive), else yes/true/1."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if positive is not None:
        return text == positive.strip().lower()
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    raise ValueError(f"can't read {value!r} as yes/no; pass the positive label explicitly")


@dataclass(frozen=True)
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def of(cls, pairs: Sequence[tuple[bool, bool]]) -> Confusion:
        """From (truth, prediction) pairs."""
        c = Counter(pairs)
        return cls(tp=c[True, True], fp=c[False, True], fn=c[True, False], tn=c[False, False])

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    # Each rate is (numerator, denominator); None when the denominator is 0 (undefined, not zero)
    def parts(self) -> dict[str, tuple[int, int]]:
        return {
            "precision": (self.tp, self.tp + self.fp),  # of what it flagged, the share that's real
            "recall": (self.tp, self.tp + self.fn),  # of what's real, the share it flagged
            "false_positive_rate": (self.fp, self.fp + self.tn),  # of the negatives, the share flagged
            "accuracy": (self.tp + self.tn, self.n),
        }

    def rate(self, name: str) -> float | None:
        k, n = self.parts()[name]
        return k / n if n else None

    def f1(self) -> float | None:
        p, r = self.rate("precision"), self.rate("recall")
        if p is None or r is None:
            return None
        return 2 * p * r / (p + r) if p + r else 0.0

    def metrics(self) -> dict[str, float | None]:
        return {**{k: self.rate(k) for k in self.parts()}, "f1": self.f1()}


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Cohen's kappa for two labellers' categories on the same items, in order.

    1 is perfect agreement, 0 is what chance alone gives. None when it's undefined (both labellers
    used one and the same category throughout, so chance agreement is already 1)."""
    if len(a) != len(b):
        raise ValueError("the two label lists must be the same length")
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def bootstrap[T](
    clusters: Sequence[Sequence[T]],
    stat: Callable[[list[T]], float | None],
    reps: int = REPS,
    seed: int = 0,
) -> tuple[float, float] | None:
    """95% percentile-bootstrap range of ``stat``, resampling whole clusters (one item each if none).

    Resamples where the statistic is undefined (e.g. no positives drawn) are skipped; None if most are."""
    groups = [list(g) for g in clusters if g]
    if not groups:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(reps):
        drawn = [x for g in rng.choices(groups, k=len(groups)) for x in g]
        v = stat(drawn)
        if v is not None:
            values.append(v)
    if len(values) < reps / 2:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[min(int(0.975 * len(values)), len(values) - 1)]


def _groups[T](items: Sequence[T], cluster_of: Sequence[Hashable] | None) -> list[list[T]]:
    if cluster_of is None:
        return [[x] for x in items]
    out: dict[Hashable, list[T]] = {}
    for x, c in zip(items, cluster_of, strict=True):
        out.setdefault(c, []).append(x)
    return list(out.values())


def confusion_report(
    pairs: Sequence[tuple[bool, bool]],
    clusters: Sequence[Hashable] | None = None,
    reps: int = REPS,
    seed: int = 0,
) -> dict:
    """Counts, metrics and 95% intervals for (truth, prediction) pairs.

    ``clusters`` (one key per pair, e.g. the task ID) switches every interval to a cluster bootstrap.
    Without it, proportions use Wilson and F1 a bootstrap over items."""
    c = Confusion.of(pairs)
    groups = _groups(list(pairs), clusters)
    intervals: dict[str, tuple[float, float] | None] = {}
    for name, (k, n) in c.parts().items():
        if clusters is None:
            intervals[name] = wilson(k, n) if n else None
        else:
            intervals[name] = bootstrap(groups, lambda xs, m=name: Confusion.of(xs).rate(m), reps, seed)
    intervals["f1"] = bootstrap(groups, lambda xs: Confusion.of(xs).f1(), reps, seed)
    return {
        "n": c.n,
        "counts": {"tp": c.tp, "fp": c.fp, "fn": c.fn, "tn": c.tn},
        "metrics": c.metrics(),
        "ci95": intervals,
        "interval_method": "cluster bootstrap" if clusters is not None else "Wilson (F1: bootstrap)",
    }


def agreement_report(
    a: Mapping[Hashable, Hashable],
    b: Mapping[Hashable, Hashable],
    clusters: Mapping[Hashable, Hashable] | None = None,
    reps: int = REPS,
    seed: int = 0,
) -> dict:
    """Two labellers' ``{item: category}`` on their shared items: agreement, kappa, disagreements."""
    shared = sorted(a.keys() & b.keys(), key=str)
    pairs = [(a[i], b[i]) for i in shared]
    keys = [clusters[i] for i in shared] if clusters is not None else None
    groups = _groups(pairs, keys)

    def kappa(xs: list[tuple[Hashable, Hashable]]) -> float | None:
        return cohen_kappa([x for x, _ in xs], [y for _, y in xs])

    def agree(xs: list[tuple[Hashable, Hashable]]) -> float | None:
        return sum(x == y for x, y in xs) / len(xs) if xs else None

    return {
        "n": len(shared),
        "only_in_a": len(a.keys() - b.keys()),
        "only_in_b": len(b.keys() - a.keys()),
        "agreement": agree(pairs),
        "agreement_ci95": bootstrap(groups, agree, reps, seed),
        "kappa": kappa(pairs),
        "kappa_ci95": bootstrap(groups, kappa, reps, seed),
        "counts_a": dict(Counter(str(x) for x, _ in pairs)),
        "counts_b": dict(Counter(str(y) for _, y in pairs)),
        "disagreements": [{"id": str(i), "a": str(a[i]), "b": str(b[i])} for i in shared if a[i] != b[i]],
    }
