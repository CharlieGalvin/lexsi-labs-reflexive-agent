"""Build the hard subset (SEL-7) from Arm A's train runs and golden trajectories v1. No API calls.

    just hard-subset          # results/episodes/arms/A-*-train.jsonl + data/golden/v1 → data/golden/hard/
    just hard-subset --force  # replace an existing build

Every train task goes in one of three groups (`reflexive.hard`): **healable** (Arm A failed, a
golden fix exists), **teacher_unsolved** (Arm A failed, no fix) or **solved_by_aya** (Arm A
passed), with the failure types the classifier finds in Arm A's conversation. Dev tasks are left
out; a test task stops the build. Golden v1 must be the build `just golden` checked.

Writes ``hard_subset.jsonl`` and ``hard_report.json`` (counts and checksums, which
`just golden-publish --tag v1.1` re-checks) to ``data/golden/hard/`` (gitignored: the data goes
to Hugging Face). The report also records the code that decided the labels: the repo commit, a
checksum of each file in `LABEL_CODE`, and any of them with uncommitted changes (publishing
refuses those: nobody could rebuild the same labels).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

from reflexive import hard, splits
from reflexive.files import ROOT, BadJsonLine, read_jsonl, sha256, shown

ARMS = ROOT / "results" / "episodes" / "arms"
GOLDEN = ROOT / "data" / "golden" / "v1"
OUT = ROOT / "data" / "golden" / "hard"
FILE = "hard_subset.jsonl"
REPORT = "hard_report.json"
GOLDEN_TRAIN = "golden_train.jsonl"
# The files that decide which records are kept, each task's group and its labels: a change to
# any of them can change the subset, so all must be committed before publishing
LABEL_CODE = (
    "src/reflexive/hallucination.py",  # the labels
    "configs/tool_schemas.json",  # the tools the classifier knows
    "src/reflexive/hard.py",  # the groups and the record format
    "src/reflexive/metrics.py",  # is_broken: which runs are refused
    "src/reflexive/splits.py",  # which tasks are train, dev or test
    "configs/splits.json",
    "scripts/data/build_hard_subset.py",  # this script: what's read and how
)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def code_version(root: Path = ROOT) -> dict[str, Any]:
    """Which code labelled the subset: the commit, the label files' checksums, and any not committed."""
    # Changed against HEAD (staged or not) or never added; plain paths, unlike `git status`'s
    # "R  old -> new" and quoted names
    changed = _git("diff", "--name-only", "HEAD", "--", *LABEL_CODE)
    untracked = _git("ls-files", "--others", "--exclude-standard", "--", *LABEL_CODE)
    unknown = changed is None or untracked is None  # not a git checkout: committed or not is unknown
    return {
        "commit": _git("rev-parse", "HEAD"),
        "files": {f: sha256(root / f)[:16] for f in LABEL_CODE},
        "uncommitted": None if unknown else sorted({*changed.splitlines(), *untracked.splitlines()}),
    }


def load_golden(folder: Path) -> tuple[list[dict[str, Any]], str]:
    """Golden v1's train records and their file's SHA-256, if they're the ones `just golden` checked."""
    path = folder / GOLDEN_TRAIN
    if not path.exists():
        raise hard.HardSubsetError(f"{shown(path)} doesn't exist: run `just golden` first")
    try:
        built = json.loads((folder / "build_report.json").read_text()).get("files") or {}
    except (OSError, ValueError, AttributeError) as e:
        raise hard.HardSubsetError(
            f"can't read {shown(folder)}/build_report.json ({type(e).__name__})"
        ) from e
    sha = sha256(path)
    if built.get(GOLDEN_TRAIN) != sha:
        raise hard.HardSubsetError(f"{shown(path)} changed since `just golden` checked it: rebuild it")
    return read_jsonl(path), sha


def load_arm(arms: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    """Arm A's train records for every domain, and where each domain's came from."""
    recs, sources = [], {}
    for d in splits.DOMAINS:
        path = arms / f"A-{d}-train.jsonl"
        if not path.exists():
            raise hard.HardSubsetError(f"{shown(path)} doesn't exist: Arm A's {d} train run (SEL-55)")
        rows = read_jsonl(path)
        wrong = {(r.get("arm"), r.get("domain")) for r in rows} - {("A", d)}
        if wrong:
            raise hard.HardSubsetError(
                f"{shown(path)} holds records of other arms or domains: {sorted(wrong)}"
            )
        recs += rows
        sources[d] = {"file": shown(path), "sha256_16": sha256(path)[:16]}
    return recs, sources


def print_table(report: dict[str, Any]) -> None:
    print(f"\n{'':10}{'healable':>10}{'unsolved':>10}{'solved':>8}   healable labels (TOOL ARG LOOP NONE)")
    for d, v in [*report["domains"].items(), ("total", report["total"])]:
        g, lab = v["groups"], v["labels"]["healable"]
        print(
            f"{d:10}{g['healable']:>10}{g['teacher_unsolved']:>10}{g['solved_by_aya']:>8}   "
            + " ".join(f"{lab[k]:>4}" for k in hard.LABELS)
        )
    left = {d: n for d, n in report["dev_left_out"].items() if n}
    if left:
        print(f"dev tasks left out: {left}")
    if report["thin"]:
        print(f"⚠ fewer than {report['min_examples']} healable tasks with: {', '.join(report['thin'])}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", type=Path, default=ARMS, help="folder with A-<domain>-train.jsonl")
    p.add_argument("--golden", type=Path, default=GOLDEN, help="golden trajectories v1 folder")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="replace an existing build")
    a = p.parse_args()

    if (a.out / FILE).exists() and not a.force:
        print(f"⛔ {shown(a.out / FILE)} exists: pass --force to rebuild it")
        return 1
    try:
        golden, golden_sha = load_golden(a.golden)
        arm, sources = load_arm(a.arms)
        records, report = hard.build(arm, golden, sources)
    except BadJsonLine as e:
        print(f"⛔ {e}")
        return 1
    except hard.HardSubsetError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / FILE).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    report = {
        "format": hard.FORMAT,
        "built": dt.date.today().isoformat(),
        "inputs": {
            "arm_a": sources,
            "golden": {"file": shown(a.golden / GOLDEN_TRAIN), "sha256": golden_sha},
        },
        "code": code_version(),
        **report,
        "files": {FILE: sha256(a.out / FILE)},
    }
    (a.out / REPORT).write_text(json.dumps(report, indent=1) + "\n")
    print(f"{len(records)} train tasks from Arm A ({', '.join(sources)}) and {len(golden)} golden fixes")
    print_table(report)
    code = report["code"]
    if code["uncommitted"]:
        print(f"⚠ uncommitted changes to {', '.join(code['uncommitted'])}: commit them before publishing")
    print(
        f"\nwrote {shown(a.out)}/{FILE} and {REPORT} (code {(code['commit'] or 'not a git checkout')[:12]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
