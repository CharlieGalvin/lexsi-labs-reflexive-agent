"""Compare two models' retention runs item by item (SEL-84, SEL-107): did healing cost anything?

    just retention-compare base sft2        # results/retention/{base,sft2}.json
    just retention-compare base sft2 --pilot

For each benchmark in configs/retention.yaml (and each Global-MMLU language), on the items both
models answered: each model's accuracy with a 95% interval, the paired difference (second − first)
with a 95% bootstrap interval over items, and the verdict by the pre-registered rule: a **drop**
only if the whole interval is below zero, a **rise** only if it's all above, otherwise **no
clear change**. Writes results/retention-<first>-vs-<second>.metrics.json. No GPU, no lm-eval:
it reads AuditKIT's saved JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from reflexive.stats import paired_bootstrap, wilson

ROOT = Path(__file__).resolve().parents[2]


def rel(path: Path) -> str:
    """A path for display: relative to the repo when it's inside it."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def load_items(path: Path) -> dict[tuple[str, str], bool]:
    """{(task, sample id): correct} from an AuditKIT RunResult JSON (unscored items left out)."""
    run = json.loads(path.read_text())
    return {
        (p["task"], str(p["sample_id"])): bool(p["correct"])
        for p in run["predictions"]
        if p.get("correct") is not None
    }


def groups(cfg: dict) -> list[tuple[str, list[str]]]:
    """(row label, task prefixes): one row per benchmark, plus one per Global-MMLU language."""
    out = []
    for b in cfg["benchmarks"]:
        out.append((b["name"], b["tasks"]))
        if len(b["tasks"]) > 1:
            out += [(f"  {t.rsplit('_', 1)[-1]}", [t]) for t in b["tasks"]]
    return out


def matches(task: str, prefixes: list[str]) -> bool:
    return any(task == t or task.startswith(t + "_") for t in prefixes)  # global_mmlu_fr_stem etc.


def compare(first: dict, second: dict, prefixes: list[str], reps: int) -> dict | None:
    shared = sorted(k for k in first.keys() & second.keys() if matches(k[0], prefixes))
    if not shared:
        return None
    a, b = {k: first[k] for k in shared}, {k: second[k] for k in shared}
    n, ka, kb = len(shared), sum(a.values()), sum(b.values())
    lo, hi = paired_bootstrap(b, a, reps=reps)  # second − first
    verdict = "drop" if hi < 0 else "rise" if lo > 0 else "no clear change"
    return {
        "items": n,
        "first": ka / n,
        "first_ci95": wilson(ka, n),
        "second": kb / n,
        "second_ci95": wilson(kb, n),
        "diff": (kb - ka) / n,
        "diff_ci95": [lo, hi],
        "verdict": verdict,
        "only_first_right": sum(a[k] and not b[k] for k in shared),
        "only_second_right": sum(b[k] and not a[k] for k in shared),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("first", help="the model before (e.g. base)")
    p.add_argument("second", help="the model after (e.g. sft2)")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "retention.yaml")
    p.add_argument("--dir", type=Path, default=ROOT / "results" / "retention")
    p.add_argument("--pilot", action="store_true", help="compare the --limit pilots instead")
    p.add_argument("--out", type=Path, help="default: results/retention-<first>-vs-<second>.metrics.json")
    a = p.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text())
    sfx = "-pilot" if a.pilot else ""
    first, second = (load_items(a.dir / f"{m}{sfx}.json") for m in (a.first, a.second))

    rows = {}
    print(
        f"{'Benchmark':<40} {'items':>5}  {a.first:>8}  {a.second:>8}"
        f"  {'difference (95% range)':>28}  verdict"
    )
    for label, prefixes in groups(cfg):
        r = compare(first, second, prefixes, cfg.get("reps", 10_000))
        if r is None:
            print(f"{label:<40} {'—':>5}  (no shared items)")
            continue
        rows[label.strip()] = r
        lo, hi = r["diff_ci95"]
        rng = f"{r['diff'] * 100:+.1f} pts ({lo * 100:+.1f} to {hi * 100:+.1f})"
        print(
            f"{label:<40} {r['items']:>5}  {r['first']:>8.1%}  {r['second']:>8.1%}  {rng:>28}  {r['verdict']}"
        )
    leftover = {
        k[0] for k in first.keys() | second.keys() if not any(matches(k[0], t) for _, t in groups(cfg))
    }
    if leftover:
        print(f"\n⚠ tasks not in the config, left out: {', '.join(sorted(leftover))}")
    out = a.out or ROOT / "results" / f"retention-{a.first}-vs-{a.second}{sfx}.metrics.json"
    by_task: dict[str, int] = defaultdict(int)
    for k in first.keys() & second.keys():
        by_task[k[0]] += 1
    out.write_text(
        json.dumps(
            {
                "first": a.first,
                "second": a.second,
                "pilot": a.pilot,
                "config": rel(a.config),
                "rule": "drop = the paired 95% interval of (second − first) lies entirely below zero",
                "method": "accuracy: Wilson; difference: paired percentile bootstrap over shared items,"
                " seed 0",
                "benchmarks": rows,
                "shared_items_by_task": dict(sorted(by_task.items())),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nsaved {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
