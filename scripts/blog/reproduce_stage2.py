"""Recompute Stage 2's headline numbers from the published conversations (no GPU or API keys).

    just reproduce-stage2                     # downloads the conversations it needs from Hugging Face
    just reproduce-stage2 --from results/episodes/arms   # or uses files you already have

Each comparison is recomputed with ``scripts/arms/stage2b_compare.py``'s own code (paired by task,
10,000 bootstrap resamples, seed 0), from the conversations in
`reagent-lexsi/reagent-tau2-conversations <https://huggingface.co/datasets/reagent-lexsi/reagent-tau2-conversations>`_,
and checked against the value the README, the blog post and ``docs/experiments/STAGE2.md`` report.
Exits 1 if any differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, read_jsonl

REPO = "reagent-lexsi/reagent-tau2-conversations"
DOMAINS = ("retail", "airline", "telecom")

# (claim, baseline, candidate, split, samples, min_sample, expected: totals, gain and 95% range)
CLAIMS: list[tuple[str, str, str, str, int, int, dict[str, Any]]] = [
    (
        "Training longer, test: 50.1% → 60.0%, +9.9 (95% +3.9 to +16.0)",
        "S2",
        "S2E4",
        "test",
        8,
        0,
        {"v2": 50.12, "candidate": 60.0, "gain": 9.88, "ci95": [3.88, 16.0]},
    ),
    (
        "The 6 samples added after the first test alone: +10.5 (+4.0 to +17.0)",
        "S2",
        "S2E4",
        "test",
        6,
        2,
        {"gain": 10.5, "ci95": [4.0, 17.0]},
    ),
    (
        "Training longer + own practice, test: +8.75 (95% +3.6 to +13.9)",
        "S2",
        "S2P1E4",
        "test",
        8,
        0,
        {"candidate": 58.88, "gain": 8.75, "ci95": [3.62, 13.88]},
    ),
    (
        "Own practice against training longer alone, test: −1.1 (95% −6.6 to +4.4)",
        "S2E4",
        "S2P1E4",
        "test",
        8,
        0,
        {"gain": -1.12, "ci95": [-6.62, 4.38]},
    ),
    (
        "Training longer, validation: 16.0 → 18.5 of 30, +2.5 (95% −0.75 to +5.88)",
        "S2",
        "S2E4",
        "val",
        8,
        0,
        {"v2": 16.0, "candidate": 18.5, "gain": 2.5, "ci95": [-0.75, 5.88]},
    ),
    (
        "FP8 against 16-bit, validation: 18.5 against 18.5 (95% −2.25 to +2.25)",
        "S2E4",
        "S2E4Q",
        "val",
        8,
        0,
        {"v2": 18.5, "candidate": 18.5, "gain": 0.0, "ci95": [-2.25, 2.25]},
    ),
]


def compare_module() -> Any:
    path = ROOT / "scripts" / "arms" / "stage2b_compare.py"
    spec = importlib.util.spec_from_file_location("stage2b_compare", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch(arm: str, split: str, src: Path | None) -> list[dict[str, Any]]:
    """One arm's conversations on one split: from ``src`` if given, else from the Hub."""
    recs: list[dict[str, Any]] = []
    for d in DOMAINS:
        name = f"{arm}-{d}-{split}.jsonl"
        if src is not None:
            path = src / name
        else:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(REPO, f"conversations/{arm}/{name}", repo_type="dataset"))
        if not path.exists():
            raise FileNotFoundError(path)
        recs += read_jsonl(path, partial_last_ok=True)
    return recs


def check(res: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """What differs from the reported value (rounded as reported, to 0.01)."""
    off = []
    for k, want in expected.items():
        got = res[k]
        if k == "ci95":
            if got is None or any(abs(g - w) > 0.005 for g, w in zip(got, want, strict=True)):
                off.append(f"95% range {got}, reported {want}")
        elif abs(got - want) > 0.005:
            off.append(f"{k} {got}, reported {want}")
    return off


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--from", dest="src", type=Path, help="use <ARM>-<domain>-<split>.jsonl files here")
    p.add_argument("--reps", type=int, default=10000)
    a = p.parse_args(argv)
    cmp = compare_module()
    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    bad = 0
    for claim, base, cand, split, samples, min_sample, expected in CLAIMS:
        for arm in (base, cand):
            if (arm, split) not in cache:
                cache[(arm, split)] = fetch(arm, split, a.src)
        res = cmp.decide(
            cmp.per_task(cache[(base, split)], min_sample),
            cmp.per_task(cache[(cand, split)], min_sample),
            samples,
            a.reps,
        )
        off = check(res, expected) + (
            [f"{len(res['incomplete'])} incomplete tasks"] if res["incomplete"] else []
        )
        ci = res["ci95"]
        got = (
            f"{res['v2']:.2f} → {res['candidate']:.2f}, {res['gain']:+.2f} (95% {ci[0]:+.2f} to {ci[1]:+.2f})"
        )
        print(f"  {'✓' if not off else '✗'} {claim}\n      recomputed: {got}  [{base} vs {cand}, {split}]")
        for o in off:
            print(f"      ⛔ {o}")
        bad += bool(off)
    print("All recomputed numbers match what we report." if not bad else f"⛔ {bad} claim(s) differ.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
