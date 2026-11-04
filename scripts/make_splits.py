"""Freeze the τ²-bench train / dev / test task IDs (SEL-36) into ``configs/splits.json``.

Run once; the output is committed and never regenerated. ``--check`` rebuilds it in memory and
confirms the committed file matches, so anyone can verify the dev pick wasn't hand-edited.

- **test**: τ²-bench v2.0.0's official test split, unchanged.
- **dev**: a random sample of the official *train* split, by task: 15 retail, 6 airline,
  15 telecom (about 20%). Seeded, and drawn from the IDs in a fixed sorted order, so it's
  reproducible.
- **train**: the official train split minus dev.

    python scripts/make_splits.py            # write configs/splits.json (refuses to overwrite)
    python scripts/make_splits.py --check    # verify the committed file
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "splits.json"
TAU2 = ROOT / "third_party" / "tau2-bench"
DEV_SIZE = {"retail": 15, "airline": 6, "telecom": 15}
SEED = "reagent-dev-v1"


def _order(ids):
    """Numeric IDs in numeric order; telecom's string IDs alphabetically."""
    return sorted(ids, key=lambda x: (0, int(x), "") if str(x).isdigit() else (1, 0, str(x)))


def build(tau2_dir: Path = TAU2) -> dict:
    domains, source = {}, {}
    for d, k in DEV_SIZE.items():
        f = tau2_dir / "data" / "tau2" / "domains" / d / "split_tasks.json"
        official = json.loads(f.read_text())
        train, test = _order(map(str, official["train"])), _order(map(str, official["test"]))
        if set(train) & set(test):
            sys.exit(f"{d}: τ²-bench's train and test overlap")
        dev = _order(random.Random(f"{SEED}:{d}").sample(train, k))
        domains[d] = {"train": [t for t in train if t not in set(dev)], "dev": dev, "test": test}
        source[d] = {"official_train": len(train), "official_test": len(test)}
    try:
        commit = subprocess.run(["git", "-C", str(tau2_dir), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "version": 1,
        "frozen": "2026-11-04",
        "source": {"benchmark": "tau2-bench v2.0.0", "commit": commit,
                   "file": "data/tau2/domains/<domain>/split_tasks.json", "counts": source},
        "dev_method": {"seed": SEED, "per_domain_seed": f"{SEED}:<domain>",
                       "sizes": DEV_SIZE, "by": "task",
                       "how": "random.Random(seed).sample(sorted official train IDs, size)"},
        "rules": [
            "test: final evaluation only. Never used for healing, teacher fixes, DSPy or any choice.",
            "dev: never trained on. For DSPy, gate checks and heal-cycle checks.",
            "train: official train minus dev. What the student trains on.",
        ],
        "domains": domains,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="verify the committed file instead of writing")
    p.add_argument("--tau2", type=Path, default=TAU2, help="τ²-bench checkout (default third_party/)")
    a = p.parse_args()
    new = build(a.tau2)
    if a.check:
        old = json.loads(OUT.read_text())
        same = old["domains"] == new["domains"]
        print("configs/splits.json matches a fresh rebuild" if same else "MISMATCH: splits.json differs")
        sys.exit(0 if same else 1)
    if OUT.exists():
        sys.exit(f"{OUT.relative_to(ROOT)} already exists: the splits are frozen. Use --check.")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(new, indent=1, ensure_ascii=False) + "\n")
    for d, s in new["domains"].items():
        print(f"{d:8s} train {len(s['train']):3d}  dev {len(s['dev']):2d}  test {len(s['test']):2d}")


if __name__ == "__main__":
    main()
