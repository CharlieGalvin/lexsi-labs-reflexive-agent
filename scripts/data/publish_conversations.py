# ruff: noqa: E501  (the data card template has long markdown lines)
"""Publish every arm's saved τ²-bench conversations to one Hugging Face dataset (SEL-110).

    just conversations-publish --dry-run              # check the files, write the card; uploads nothing
    just conversations-publish                        # upload every arm under results/episodes/arms/
    just conversations-publish --arms S2E4 S2P1E4     # only these arms (e.g. from the pod that ran them)

One dataset holds the conversations of every arm on every split, so anyone can recompute our
numbers without a GPU or API keys: ``conversations/<ARM>/<ARM>-<domain>-<split>.jsonl``, the frozen
``splits.json``, a ``manifest.json`` with each file's counts and SHA-256, and the data card.

The arms' files live on different machines (Stage 1 on a laptop, Stage 2 on the pods), so run it
on each: an upload adds and replaces only the files it sends, and merges its entries into the
manifest already on the Hub. Before uploading it checks every record: valid JSON, the domain and
split in the file's name match the task's split in ``configs/splits.json`` (``val``, ``fresh`` and
``trainpool`` are slices of train), and no line looks like an API key. The repo is created
private; make it public on the Hub when the submission goes in.

Uses your own Hugging Face token: ``HF_TOKEN`` in ``.env``, or ``hf auth login``.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import re
import shutil
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import ROOT, read_jsonl, sha256, shown

REPO = "reagent-lexsi/reagent-tau2-conversations"
SRC = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "data" / "conversations"
NAME = re.compile(r"^(?P<arm>[A-Z][A-Za-z0-9]*)-(?P<domain>retail|airline|telecom)-(?P<split>[a-z]+)\.jsonl$")
SLICES_OF_TRAIN = {"val", "fresh", "trainpool"}
SECRET = re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|rpa_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{30,})")

ARMS = {
    "A": "Base Aya Expanse 32B through Cohere's API (Stage 1, Arm A)",
    "C": "Arm A with a fixed instruction not to hallucinate (Stage 1, Arm C)",
    "P": "Base Aya Expanse 32B on our own vLLM server",
    "S": "SFT v1",
    "SL": "SFT v1's data trained longer (a control)",
    "S2": "SFT v2, the healed model (Stage 1, Arm B)",
    "S3": "SFT v3: v2's data plus 20 verified retail teacher conversations",
    "S4": "SFT v4: v2's data plus 29 teacher fixes at v2's own wrong decisions",
    "S2G": "SFT v2 with the consent and mismatch guard",
    "S2E4": "SFT v2's data trained for 4 epochs (the final model)",
    "S2E4Q": "S2E4 served with FP8 weights and a 64K context",
    "GPT": "GPT-4.1 as the agent, a reference (OpenRouter, 2 samples per task)",
    "S2E4S1": "S2E4's recipe trained again with seed 1",
    "S2PR": "SFT v2 practising on train tasks at temperature 0.7: the source of S2P1's practice data",
    "S2P1": "SFT v2's data plus 100 of its own verified practice conversations, 2 epochs",
    "S2P1E4": "S2P1's data, 4 epochs",
}


class CheckError(ValueError):
    """A file isn't fit to publish."""


def check(path: Path) -> dict[str, Any]:
    """One file's manifest entry, after checking every record. Raises CheckError."""
    m = NAME.match(path.name)
    if not m:
        raise CheckError(f"{path.name}: not <ARM>-<domain>-<split>.jsonl")
    arm, domain, split = m["arm"], m["domain"], m["split"]
    base_split = "train" if split in SLICES_OF_TRAIN else split
    if base_split not in splits.SPLITS:
        raise CheckError(f"{path.name}: unknown split {split!r}")
    text = path.read_text()
    if SECRET.search(text):
        raise CheckError(f"{path.name}: a line looks like an API key or token; not uploading it")
    recs = read_jsonl(path)
    if not recs:
        raise CheckError(f"{path.name}: no conversations")
    tasks: collections.Counter[str] = collections.Counter()
    passed = crashed = 0
    dev_tasks: set[str] = set()
    for i, r in enumerate(recs, 1):
        task = str(r.get("task_id"))
        try:
            actual = splits.split_of(domain, task)
        except KeyError:
            raise CheckError(f"{path.name} line {i}: {domain} task {task!r} is in no split") from None
        if actual == "dev" and split == "train":
            dev_tasks.add(task)  # early runs used τ²-bench's own train split: our train + dev
        elif actual != base_split:
            raise CheckError(f"{path.name} line {i}: task {task!r} is a {actual} task, not {base_split}")
        tasks[task] += 1
        if r.get("error") or r.get("termination") == "error":
            crashed += 1
        elif (r.get("outcome") or {}).get("passed"):
            passed += 1
    scored = len(recs) - crashed
    return {
        "file": f"conversations/{arm}/{path.name}",
        "arm": arm,
        "domain": domain,
        "split": split,
        "conversations": len(recs),
        "tasks": len(tasks),
        "max_samples_per_task": max(tasks.values()),
        "crashed": crashed,
        "passed": passed,
        "success": round(passed / scored, 4) if scored else None,
        "dev_tasks_in_train_file": len(dev_tasks),
        "sha256": sha256(path),
    }


def card(manifest: dict[str, dict[str, Any]]) -> str:
    """The data card, from the whole manifest (this upload's files and those already on the Hub)."""
    by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for e in manifest.values():
        by_arm[e["arm"]].append(e)
    rows = []
    for arm in sorted(by_arm, key=lambda a: (list(ARMS).index(a) if a in ARMS else len(ARMS), a)):
        es = by_arm[arm]
        splits_ = ", ".join(sorted({e["split"] for e in es}))
        n = sum(e["conversations"] for e in es)
        rows.append(
            f"| {arm} | {ARMS.get(arm, 'see docs/experiments in the code repo')} | {splits_} | {n:,} |"
        )
    files = "\n".join(
        f"| `{e['file']}` | {e['conversations']} | {e['tasks']} | {e['max_samples_per_task']} | {e['crashed']} | "
        + (f"{100 * e['success']:.1f}%" if e["success"] is not None else "–")
        + " |"
        for e in sorted(manifest.values(), key=lambda e: e["file"])
    )
    return f"""---
license: cc-by-nc-4.0
language: [en]
pretty_name: ReAgent τ²-bench conversations
tags: [agents, tool-use, tau2-bench, aya, self-healing]
configs:
- config_name: default
  data_files: "conversations/*/*.jsonl"
---

# ReAgent: every arm's τ²-bench conversations

Every conversation behind the results of [ReAgent](https://github.com/CharlieGalvin/lexsi-labs-reflexive-agent),
a self-healing loop for Aya Expanse 32B on [τ²-bench](https://github.com/sierra-research/tau2-bench) v2.0.0
(retail, airline, telecom). Each line is one conversation: the messages and tool calls, τ²-bench's verdict
(`outcome.passed`), and how it ended. The customer is simulated by GPT-4.1.

**Splits** are frozen in `splits.json` (also `configs/splits.json` in the code): train 142 tasks, dev 36,
test 100 (τ²-bench's own test split). `val`, `fresh` and `trainpool` are slices of train. Some early
`-train` files ran τ²-bench's own train split, which is our train and dev together: `manifest.json` counts
their dev tasks (`dev_tasks_in_train_file`). Test conversations were never trained on. Every file was
checked against the splits before upload.

## Arms

| Arm | What it is | Splits | Conversations |
|---|---|---|---:|
{chr(10).join(rows)}

## Files

Success excludes conversations that crashed (rate limits, server errors); those were re-run, and the re-run
is in the file too. `manifest.json` has each file's SHA-256.

| File | Conversations | Tasks | Samples per task (max) | Crashed | Success |
|---|---:|---:|---:|---:|---:|
{files}

## Licence

CC BY-NC 4.0: the agent's replies come from Aya Expanse 32B (CC BY-NC 4.0). τ²-bench's tasks and policies
are MIT-licensed by Sierra.

Updated {dt.date.today().isoformat()}.
"""


def remote_manifest(repo: str) -> dict[str, dict[str, Any]]:
    """The manifest already on the Hub, or an empty one if the repo or the file is new."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(repo, "manifest.json", repo_type="dataset")).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError):
        return {}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", help="only these arms (default: every arm found)")
    p.add_argument("--src", type=Path, default=SRC, help=f"default: {shown(SRC)}")
    p.add_argument("--repo", default=REPO)
    p.add_argument(
        "--dry-run", action="store_true", help="check and write data/conversations/, upload nothing"
    )
    a = p.parse_args(argv)

    paths = sorted(f for f in a.src.glob("*.jsonl") if NAME.match(f.name))
    if a.arms:
        paths = [f for f in paths if NAME.match(f.name)["arm"] in a.arms]
    if not paths:
        print(
            f"⛔ no <ARM>-<domain>-<split>.jsonl files in {shown(a.src)}"
            + (f" for {a.arms}" if a.arms else "")
        )
        return 1

    local, bad = {}, []
    for f in paths:
        try:
            e = check(f)
            local[e["file"]] = e
        except (CheckError, ValueError) as err:
            bad.append(str(err))
    for b in bad:
        print(f"⛔ {b}")
    if bad:
        print("nothing written: fix or move the files above, or leave their arm out with --arms")
        return 1
    for e in local.values():
        rate = f"{100 * e['success']:.1f}%" if e["success"] is not None else "–"
        print(
            f"   ✓ {e['file']}: {e['conversations']} conversations, {e['tasks']} tasks, {e['crashed']} crashed, {rate}"
        )

    api = None
    manifest = dict(local)
    if not a.dry_run:
        from huggingface_hub import HfApi

        api = HfApi()
        manifest = {**remote_manifest(a.repo), **local}

    shutil.rmtree(OUT, ignore_errors=True)
    for e in local.values():
        dest = OUT / e["file"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.src / Path(e["file"]).name, dest)
    shutil.copy(splits.SPLITS_PATH, OUT / "splits.json")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    (OUT / "README.md").write_text(card(manifest))
    print(f"wrote {shown(OUT)}: {len(local)} files, splits.json, manifest.json, README.md")
    if a.dry_run:
        print("(dry run: nothing uploaded)")
        return 0

    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    arms = sorted({e["arm"] for e in local.values()})
    info = api.upload_folder(
        folder_path=str(OUT),
        repo_id=a.repo,
        repo_type="dataset",
        commit_message=f"Conversations for arms {', '.join(arms)}",
    )
    print(f"uploaded to https://huggingface.co/datasets/{a.repo} at {info.oid[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
