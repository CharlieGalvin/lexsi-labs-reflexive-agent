"""Score a detector against a reference, or two labellers against each other (SEL-85, SEL-98).

    # A detector's flags against the truth: counts, precision, recall, F1, false-positive rate
    just label-metrics confusion results/labels/checker-dev-val.csv --truth failed --pred flagged \
        --cluster task_id --json results/checker-vs-tau2.metrics.json

    # Two labellers on the same items: agreement and Cohen's kappa, and the items to settle
    just label-metrics agreement labels/katie.csv labels/charlie.csv --label label --json out.json

Input: CSV or JSONL (by extension), one row per item. Yes/no columns accept yes/no, true/false or
1/0, or name the positive value with ``--positive`` (e.g. ``--positive fail``). ``--cluster``
names a column (e.g. the task ID) whose rows belong together, such as a task's two samples: the
intervals then resample whole clusters. No API calls, no GPU.

For the classifier audit (SEL-85), run ``confusion`` on each sample separately: precision comes
from the flagged sample (every row predicted positive), recall from the random sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from reflexive.labels import agreement_report, as_bool, confusion_report


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def need(rows: list[dict], *cols: str, path: Path) -> None:
    missing = [c for c in cols if rows and c not in rows[0]]
    if missing:
        raise SystemExit(f"{path}: no column {', '.join(missing)} (columns: {', '.join(rows[0])})")


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def fmt_ci(ci: tuple[float, float] | list[float] | None, pct: bool = True) -> str:
    if ci is None:
        return ""
    lo, hi = ci
    return f"({lo:.1%} to {hi:.1%})" if pct else f"({lo:.2f} to {hi:.2f})"


def confusion(a: argparse.Namespace) -> dict:
    rows = read_rows(a.file)
    need(rows, a.truth, a.pred, *([a.cluster] if a.cluster else []), path=a.file)
    pairs = [(as_bool(r[a.truth], a.positive), as_bool(r[a.pred], a.pred_positive)) for r in rows]
    clusters = [r[a.cluster] for r in rows] if a.cluster else None
    rep = confusion_report(pairs, clusters, reps=a.reps)
    c = rep["counts"]
    print(f"{a.file}: {rep['n']} items; truth = {a.truth}, prediction = {a.pred}\n")
    print("                 predicted yes   predicted no")
    print(f"  truly yes      TP {c['tp']:>10}   FN {c['fn']:>9}")
    print(f"  truly no       FP {c['fp']:>10}   TN {c['tn']:>9}\n")
    labels = {
        "precision": "Precision (of flagged, truly yes)",
        "recall": "Recall (of truly yes, flagged)",
        "f1": "F1",
        "false_positive_rate": "False-positive rate (of truly no, flagged)",
        "accuracy": "Accuracy",
    }
    for k, label in labels.items():
        print(f"  {label:<44} {fmt(rep['metrics'][k]):>7}  {fmt_ci(rep['ci95'][k])}")
    print(f"\n  95% intervals: {rep['interval_method']}")
    return {"file": str(a.file), "truth": a.truth, "prediction": a.pred, **rep}


def agreement(a: argparse.Namespace) -> dict:
    ra, rb = read_rows(a.file_a), read_rows(a.file_b)
    need(ra, a.id, a.label, path=a.file_a)
    need(rb, a.id, a.label, path=a.file_b)
    la = {r[a.id]: r[a.label].strip().lower() for r in ra}
    lb = {r[a.id]: r[a.label].strip().lower() for r in rb}
    clusters = None
    if a.cluster:
        need(ra, a.cluster, path=a.file_a)
        clusters = {r[a.id]: r[a.cluster] for r in ra}
    rep = agreement_report(la, lb, clusters, reps=a.reps)
    print(
        f"{a.file_a.name} against {a.file_b.name}: {rep['n']} shared items "
        f"({rep['only_in_a']} only in the first, {rep['only_in_b']} only in the second)\n"
    )
    print(f"  Raw agreement  {fmt(rep['agreement']):>7}  {fmt_ci(rep['agreement_ci95'])}")
    kappa = "n/a" if rep["kappa"] is None else f"{rep['kappa']:.2f}"
    print(f"  Cohen's kappa  {kappa:>7}  {fmt_ci(rep['kappa_ci95'], pct=False)}")
    cats = sorted(set(rep["counts_a"]) | set(rep["counts_b"]))
    print("\n  Category           first  second")
    for k in cats:
        print(f"  {k:<18} {rep['counts_a'].get(k, 0):>5}  {rep['counts_b'].get(k, 0):>6}")
    if rep["disagreements"]:
        print(f"\n  To settle ({len(rep['disagreements'])}):")
        for d in rep["disagreements"]:
            print(f"    {d['id']}: {d['a']} / {d['b']}")
    return {"first": str(a.file_a), "second": str(a.file_b), "label": a.label, **rep}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("confusion", help="a detector's flags against the truth")
    c.add_argument("file", type=Path)
    c.add_argument("--truth", required=True, help="column with the reference yes/no")
    c.add_argument("--pred", required=True, help="column with the detector's yes/no")
    c.add_argument("--positive", help="the truth column's 'yes' value, if not yes/true/1")
    c.add_argument("--pred-positive", help="the prediction column's 'yes' value, if not yes/true/1")
    g = sub.add_parser("agreement", help="two labellers on the same items")
    g.add_argument("file_a", type=Path)
    g.add_argument("file_b", type=Path)
    g.add_argument("--id", default="id", help="column identifying the item (default: id)")
    g.add_argument("--label", required=True, help="column with the category")
    for s in (c, g):
        s.add_argument("--cluster", help="column grouping rows that belong together (e.g. task_id)")
        s.add_argument("--reps", type=int, default=10_000, help="bootstrap resamples (default 10,000)")
        s.add_argument("--json", type=Path, help="also save the report as JSON here")
    a = p.parse_args(argv)
    rep = confusion(a) if a.cmd == "confusion" else agreement(a)
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=2) + "\n")
        print(f"\nsaved {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
