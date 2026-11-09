"""SFT v2 data (SEL-71): v1's examples plus the teacher's extra verified passes, for some domains.

    python scripts/data/build_extra_fixes.py --domains airline          # → data/train/sft-v2.jsonl
    python scripts/data/build_extra_fixes.py --domains airline --dry-run

Reads the teacher's fix run (``results/fixes/glm53f.jsonl``, after
``scripts/teacher/fixes.py --domains airline --tasks all --all-attempts``) and keeps the extra
conversations: clean passes (`golden.why_not_clean`), on **train** tasks, in the given domains,
whose task already has a v1 example in the ``healable`` group (so the held-out ``solved_by_aya``
tasks stay unseen), and that aren't the attempt v1 already uses or a word-for-word repeat of a
kept one. Each is converted exactly as golden v1 was (`golden.to_record`, read back with
`golden.check_record`) and checked as a training example (`train_data.problems_of`); one failure
stops the build and writes nothing.

Writes the v1 examples unchanged plus the extras (group ``extra``) to ``--out``, and a report
next to it (counts per domain and per task). Never reads or writes a dev or test task.
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
from pathlib import Path
from typing import Any

from reflexive import golden, splits
from reflexive.files import ROOT, read_jsonl, sha256, shown
from reflexive.train_data import FORMAT, TrainDataError, problems_of

FIXES = ROOT / "results" / "fixes" / "glm53f.jsonl"
BASE = ROOT / "data" / "train" / "sft-all.jsonl"
OUT = ROOT / "data" / "train" / "sft-v2.jsonl"
Record = dict[str, Any]


def _build_golden() -> Any:
    """scripts/data/build_golden.py, for `prompts_and_tools` (one system prompt per domain)."""
    spec = importlib.util.spec_from_file_location(
        "build_golden", ROOT / "scripts" / "data" / "build_golden.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def attempt_id(r: Record) -> str:
    """``<teacher>/<domain>/<task>/<attempt>``: the id golden records (and v2's examples) carry."""
    return f"{r.get('sample_model', 'teacher')}/{r.get('domain')}/{r.get('task_id')}/{r.get('attempt') or 1}"


def candidates(
    fixes: list[Record], base: list[Record], domains: set[str]
) -> tuple[list[Record], collections.Counter]:
    """The extra passes worth converting, and why the rest were left out."""
    healable = {(r["domain"], str(r["task_id"])) for r in base if r["group"] == "healable"}
    used = {r["source"]["golden_id"] for r in base}  # "<teacher>/<domain>/<task>/<attempt>"
    why: collections.Counter = collections.Counter()
    last: dict[str, Record] = {}  # a re-run of the teacher can append the same attempt again
    for r in fixes:
        if r.get("domain") in domains:
            if attempt_id(r) in last:
                why["an earlier record of the same attempt (the last one is kept)"] += 1
            last[attempt_id(r)] = r
    keep = []
    for r in last.values():
        d, t = str(r["domain"]), str(r.get("task_id"))  # every record here is in ``domains``
        try:
            split = splits.split_of(d, t)
        except KeyError:
            split = "no split"
        if split != "train":
            why[f"a {split} task"] += 1
            continue
        if (d, t) not in healable:
            why["task not in v1's healable group (held out, or no v1 fix)"] += 1
            continue
        if attempt_id(r) in used:
            why["the attempt v1 already uses"] += 1
            continue
        if (problem := golden.why_not_clean(r)) is not None:
            why[f"not a clean pass: {problem}"] += 1
            continue
        keep.append(r)
    return keep, why


def to_example(rec: Record, system: tuple[str, list[dict]]) -> Record:
    sp, tools = system
    g = golden.to_record(
        rec, sp, tools, "train", source={"sel": "SEL-71", "attempt": rec.get("attempt") or 1}
    )
    if bad := golden.check_record(g, golden.agent_view(rec["messages"], sp)):
        raise TrainDataError(f"{g['id']}: {'; '.join(bad)}")
    if bad := problems_of(g):
        raise TrainDataError(*bad)
    return {
        "id": g["id"],
        "format": FORMAT,
        "domain": g["domain"],
        "task_id": str(g["task_id"]),
        "group": "extra",
        "labels": ["sel-71"],
        "messages": [{"role": m["role"], "content": m["content"]} for m in g["messages"]],
        "train_on": list(g["train_on"]),
        "source": {"golden_id": g["id"], "hard_id": None, "teacher": g.get("teacher")},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domains", nargs="+", required=True, choices=["retail", "airline", "telecom"])
    p.add_argument("--fixes", type=Path, default=FIXES)
    p.add_argument("--base", type=Path, default=BASE, help="v1's examples (just train-data --select all)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    a = p.parse_args()

    from loguru import logger

    logger.remove()
    base = read_jsonl(a.base)
    fixes = read_jsonl(a.fixes, partial_last_ok=True)
    keep, why = candidates(fixes, base, set(a.domains))
    system = {
        d: _build_golden().prompts_and_tools(d, next(str(r["task_id"]) for r in keep if r["domain"] == d))
        for d in {r["domain"] for r in keep}
    }
    try:
        extras, seen = [], {json.dumps([m["content"] for m in r["messages"]]) for r in base}
        for r in keep:
            ex = to_example(r, system[r["domain"]])
            text = json.dumps([m["content"] for m in ex["messages"]])
            if text in seen:
                why["word-for-word repeat"] += 1
                continue
            seen.add(text)
            extras.append(ex)
    except TrainDataError as e:
        for line in e.lines:
            print(f"⛔ {line}")
        return 1

    per_domain = collections.Counter(r["domain"] for r in extras)
    per_task = collections.Counter(f"{r['domain']}:{r['task_id']}" for r in extras)
    v1 = collections.Counter(r["domain"] for r in base if r["group"] == "healable")
    print(f"extra verified passes kept: {len(extras)} ({dict(per_domain)}) from {len(set(per_task))} tasks")
    for d in sorted(per_domain):
        print(f"  {d}: {v1[d]} v1 examples → {v1[d] + per_domain[d]}")
    for reason, n in why.most_common():
        print(f"  left out: {n} × {reason}")
    if a.dry_run or not extras:
        return 0 if extras or a.dry_run else 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in [*base, *extras]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "sel": "SEL-71",
        "base": {"file": shown(a.base), "sha256": sha256(a.base), "examples": len(base)},
        "fixes": {"file": shown(a.fixes), "sha256": sha256(a.fixes)},
        "domains": a.domains,
        "extra": {"total": len(extras), "per_domain": dict(per_domain), "per_task": dict(per_task)},
        "left_out": dict(why),
    }
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"→ {shown(a.out)} ({len(base)} + {len(extras)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
