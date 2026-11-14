"""Tool / Arg / Loop counts per arm, domain and sample, by the classifier (no API calls).

    just arm-tool-arg H --split val          # SEL-75's rule, part 3: Tool or Arg in ≤ 4 of 60
    just arm-tool-arg S2 H                   # dev (default), two arms

Counts conversations with any Tool or Arg finding (`reflexive.hallucination.classify_record`, as
`reflexive.compare` and SFT.md count it: v2 had 4 of 60 on validation), and with a Loop finding.
Reads ``results/episodes/arms/<arm>-<domain>-<split>.jsonl``; the last record per task and sample
counts, and crashed conversations are left out (shown separately).
"""

from __future__ import annotations

import argparse
import collections
import sys

from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.hallucination import FailureType, classify_record
from reflexive.metrics import is_broken

DOMAINS = ("retail", "airline", "telecom")


def counts(arm: str, split: str) -> dict[str, list[int]]:
    """{domain: [tool_or_arg, loop, scored, crashed]}"""
    out: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    for d in DOMAINS:
        path = ROOT / "results" / "episodes" / "arms" / f"{arm}-{d}-{split}.jsonl"
        if not path.exists():
            continue
        for r in latest_records(read_jsonl(path, partial_last_ok=True)):
            if is_broken(r):
                out[d][3] += 1
                continue
            types = classify_record(r).types
            out[d][0] += bool(types & {FailureType.TOOL, FailureType.ARG})
            out[d][1] += FailureType.LOOP in types
            out[d][2] += 1
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arms", nargs="+")
    p.add_argument("--split", default="dev")
    a = p.parse_args(argv)
    for arm in a.arms:
        c = counts(arm, a.split)
        if not c:
            print(f"{arm}: no {a.split} files")
            continue
        tot = [sum(v[i] for v in c.values()) for i in range(4)]
        print(
            f"{arm} ({a.split}): Tool or Arg in {tot[0]}/{tot[2]} conversations, Loop in {tot[1]}/{tot[2]}"
            + (f"; {tot[3]} crashed, not counted" if tot[3] else "")
        )
        for d in DOMAINS:
            if d in c:
                v = c[d]
                print(
                    f"  {d:8s} Tool or Arg {v[0]}/{v[2]}, Loop {v[1]}/{v[2]}"
                    + (f", {v[3]} crashed" if v[3] else "")
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
