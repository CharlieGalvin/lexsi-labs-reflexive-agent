"""Summarise an arm's rollout file as a ``.metrics.json`` (SEL-55). No API calls.

    just arm-metrics results/episodes/arms/A-retail-train.jsonl
    just arm-metrics results/episodes/arms/A-telecom-train.jsonl --only dev   # its dev tasks only
    python scripts/arms/metrics.py <file> --note "…"   # a note with spaces or brackets: `just` drops quotes

Writes ``results/arm-<arm>-<domain>-<split>.metrics.json`` (committed: `results/` keeps
``*.metrics.json``), in the format of Chiara's first ones (4 Nov). The numbers are defined in
``reflexive.metrics``.

**The split comes from the records**, never the file name: each task is looked up in the
frozen splits (``configs/splits.json``). A file holding both train and dev tasks (τ²-bench's own
train split, which an older run used) needs ``--only train`` or ``--only dev``. The script
refuses, before writing anything: a record without a domain or task ID, an unknown domain, a
test-split task anywhere in the file (whatever ``--only`` says), a task in no split, several
domains or arms in one file, a file name (``<arm>-<domain>-<split>``) whose arm, domain or split
disagrees with its records, a selection (after ``--only``) in which every conversation
crashed, and replacing an existing summary unless ``--force`` is given. A half-written last
line (a stopped rollout) is skipped, as ``rollout.py`` does when it resumes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import ROOT, BadJsonLine, read_jsonl, shown
from reflexive.metrics import COST_NOTE, arm_metrics, is_broken

COHERE_API = "https://api.cohere.ai/compatibility/v1"
_NAME = re.compile(r"^([A-Za-z0-9]+)-([a-z]+)-")  # A-retail-train(.retry) → arm A, domain retail


def student(rec: dict[str, Any]) -> str | None:
    """The student as the record describes it: model, endpoint, mode and prompt style.

    Records from before 6 Nov don't save the endpoint. They all ran on Cohere's API
    (configs/arms.yaml at the time), which the description then says it assumes."""
    model = rec.get("agent_llm")
    if not model:
        return None
    base = rec.get("agent_api_base")
    where = "Cohere API" if base == COHERE_API else base or "Cohere API, assumed: the records don't say"
    mode, style = rec.get("agent_mode", "prompted"), rec.get("agent_style", "cohere")
    return (
        f"{model} ({where}, {mode}, {style} Action: format)"
        if mode == "prompted"
        else f"{model} ({where}, {mode})"
    )


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Refused(Exception):
    """The file can't be summarised as asked; the message says why and what to do."""


def _one_run(recs: list[dict[str, Any]], path: Path) -> str:
    """The run's domain, if the file is one arm on one known domain and its name agrees."""
    for i, r in enumerate(recs, start=1):
        if not r.get("domain") or r.get("task_id") is None:
            raise Refused(f"{shown(path)}, record {i} has no domain or task_id: is it a rollout file?")
    for key in ("domain", "arm"):
        found = {r.get(key) for r in recs}
        if len(found) > 1:
            raise Refused(
                f"{shown(path)} mixes several {key}s ({', '.join(sorted(map(str, found)))}): one run per file"
            )
    domain, arm = recs[0]["domain"], recs[0].get("arm")
    known = splits.load()["domains"]
    if domain not in known:
        raise Refused(f"unknown domain {domain!r}: configs/splits.json has {', '.join(sorted(known))}")
    named = _NAME.match(path.stem)
    if named and named.group(2) in known:  # a name like ours, not some custom one
        for what, in_name, in_recs in (("arm", named.group(1), arm), ("domain", named.group(2), domain)):
            if in_name != str(in_recs):
                raise Refused(
                    f"{shown(path)} is named {what} {in_name}, but its records are {what} {in_recs}"
                )
    return domain


def select(recs: list[dict[str, Any]], path: Path, only: str | None) -> tuple[list[dict[str, Any]], str]:
    """The records to summarise and their split, or ``Refused``. Checks run before ``--only``,
    except the all-crashed one, which looks at what ``--only`` kept."""
    domain = _one_run(recs, path)
    split_of: dict[int, str] = {}
    for i, r in enumerate(recs):
        try:
            split_of[i] = splits.split_of(domain, str(r["task_id"]))
        except KeyError:
            raise Refused(
                f"{domain} task {r['task_id']!r} is in no frozen split (configs/splits.json)"
            ) from None
    test = [str(recs[i]["task_id"]) for i, sp in split_of.items() if sp == "test"]
    if test:
        raise Refused(
            f"{len(test)} test-split tasks in {shown(path)} (e.g. {test[0]}): summarise test runs separately"
        )
    if only:
        recs = [r for i, r in enumerate(recs) if split_of[i] == only]
        if not recs:
            raise Refused(f"no {only} tasks in {shown(path)}")
        split = only
    else:
        found = sorted(set(split_of.values()))
        if len(found) > 1:
            raise Refused(f"{shown(path)} holds {' and '.join(found)} tasks: pass --only train or --only dev")
        split = found[0]
        in_name = path.stem.rsplit("-", 1)[-1]  # A-retail-train → train
        if in_name in splits.SPLITS and in_name != split:
            raise Refused(f"{shown(path)} is named {in_name}, but its tasks are {split} tasks")
    if all(is_broken(r) for r in recs):
        kept = f"the {only} tasks of {shown(path)}" if only else shown(path)
        raise Refused(f"every conversation in {kept} crashed: re-run the arm, then summarise it")
    return recs, split


def header(
    recs: list[dict[str, Any]], path: Path, split: str, only: str | None, owner: str
) -> dict[str, Any]:
    """Who ran what, where: the fields before the numbers."""
    first = recs[0]
    arm, domain = first.get("arm", "?"), first["domain"]
    scored = [r for r in recs if not is_broken(r)]
    judged = sum(bool(r["outcome"].get("judged")) for r in scored)
    skipped = sum(bool(r["outcome"].get("skipped")) for r in scored)
    samples = max(r.get("sample", 0) for r in recs) + 1
    return {
        "run": f"arm-{arm}-{domain}-{split}",
        "date": dt.date.today().isoformat(),
        "owner": owner,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "config": "configs/arms.yaml",
        "arm": arm,
        "prompt_variant": first.get("prompt_variant"),
        "prompt_version": first.get("prompt_version"),
        "student": student(first),
        "customer": first.get("user_llm"),
        "domain": domain,
        "split": f"{split} (frozen splits, configs/splits.json)",
        "samples_per_task": samples,
        "judges": bool(judged),
        "note_judges": (
            "telecom has no LLM judges: scores are complete"
            if domain == "telecom"
            else f"LLM-judged checks skipped in {skipped} of {len(scored)} conversations: "
            "those scores use the deterministic checks only"
            if skipped
            else f"τ²-bench's LLM judges ran where a score needed one ({judged} of {len(scored)} "
            "conversations; the others ended early or have no judged check): scores are complete"
        ),
        "note_cost": COST_NOTE,
        "records": f"{shown(path)} (not committed)" + (f", the {only} tasks only" if only else ""),
        **(
            {"note_samples": "the 95% range counts every conversation as independent: too narrow"}
            if samples > 1
            else {}
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "records", type=Path, help="a rollout file, results/episodes/arms/<arm>-<domain>-<split>.jsonl"
    )
    p.add_argument("--only", choices=["train", "dev"], help="keep only this frozen split's tasks")
    p.add_argument("--owner", default=_git("config", "user.name"), help="who ran it (default: git user.name)")
    p.add_argument("--note", help="anything a reader should know, saved as `history`")
    p.add_argument("--out", type=Path, help="default: results/arm-<arm>-<domain>-<split>.metrics.json")
    p.add_argument("--force", action="store_true", help="replace an existing summary")
    a = p.parse_args()

    try:
        recs = read_jsonl(a.records, partial_last_ok=True)
    except (BadJsonLine, FileNotFoundError) as e:
        print(f"⛔ {e}")
        return 1
    if not recs:
        print(f"⛔ {shown(a.records)} has no records")
        return 1
    try:
        recs, split = select(recs, a.records, a.only)
    except Refused as e:
        print(f"⛔ {e}")
        return 1

    out = {**header(recs, a.records, split, a.only, a.owner), **arm_metrics(recs)}
    if a.note:
        out["history"] = a.note
    path = a.out or ROOT / "results" / f"{out['run']}.metrics.json"
    if path.exists() and not a.force:
        print(f"⛔ {shown(path)} exists: pass --force to replace it (it may hold someone's notes)")
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    lo, hi = out["task_success_wilson95"]
    print(
        f"{out['run']}: {out['passed']}/{out['conversations']} passed ({lo:.0%}–{hi:.0%}), "
        f"{out['crashed']} crashed, ${out['cost_usd']['total']:.2f} → {shown(path)}"
    )
    if out["crashed"]:
        print("⚠ crashed conversations aren't scored: re-run the arm to retry them, then summarise again")
    return 0


if __name__ == "__main__":
    sys.exit(main())
