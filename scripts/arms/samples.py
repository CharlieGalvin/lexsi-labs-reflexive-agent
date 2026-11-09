"""Pass counts per arm, domain and sample: how much a result moves from run to run.

    just arm-samples P S                      # dev (default)
    just arm-samples S P --split val          # the validation files (<arm>-<domain>-val.jsonl)

For each arm: passes per domain for each sample, the totals per sample, and the mean. Uses
the runs' saved verdicts (free); re-score with ``just oracle-score`` for the reported numbers.
The last record per task and sample counts, so a re-run replaces its crash; crashed
conversations (``reflexive.metrics.is_broken``) are counted separately.
"""

from __future__ import annotations

import argparse
import collections
import glob
from pathlib import Path

from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.metrics import is_broken

DOMAINS = ("retail", "airline", "telecom")


def table(arm: str, split: str) -> dict[tuple[str, int], list[int]]:
    """{(domain, sample): [passed, scored, crashed]}"""
    out: dict[tuple[str, int], list[int]] = collections.defaultdict(lambda: [0, 0, 0])
    for d in DOMAINS:
        for f in glob.glob(str(ROOT / "results" / "episodes" / "arms" / f"{arm}-{d}-{split}.jsonl")):
            for r in latest_records(read_jsonl(Path(f), partial_last_ok=True)):
                c = out[(d, r.get("sample", 0))]
                if is_broken(r):
                    c[2] += 1
                else:
                    c[0] += bool((r.get("outcome") or {}).get("passed"))
                    c[1] += 1
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+")
    p.add_argument("--split", default="dev", help="the file suffix: dev, val, test, train")
    a = p.parse_args()
    for arm in a.arms:
        t = table(arm, a.split)
        samples = sorted({s for _, s in t})
        if not samples:
            print(f"{arm}: no {a.split} files")
            continue
        print(f"{arm} ({a.split})")
        for d in DOMAINS:
            row = "  ".join(f"s{s} {t[(d, s)][0]}/{t[(d, s)][1]}" for s in samples if (d, s) in t)
            print(f"  {d:<8} {row}")
        totals = [(sum(t[(d, s)][0] for d in DOMAINS), sum(t[(d, s)][1] for d in DOMAINS)) for s in samples]
        crashed = sum(t[k][2] for k in t)
        mean = sum(k for k, _ in totals) / len(totals)
        print(
            "  total    "
            + "  ".join(f"s{s} {k}/{n}" for s, (k, n) in zip(samples, totals, strict=True))
            + f"   mean {mean:.1f}"
            + (f"   ({crashed} crashed, left out)" if crashed else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
