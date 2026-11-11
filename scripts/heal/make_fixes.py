"""Teacher fixes at the student's Tool / Arg / Loop failures, for heal GRPO (SEL-75).

    just heal-fixes --dry-run                 # the heal points, by type: no API call
    just heal-fixes --limit 5                 # a smoke run: 5 points
    just heal-fixes                           # every point (~210 from Arm A; resumes)
    just heal-fixes --episodes results/episodes/arms/S2-*-trainpool.jsonl   # + SFT v2's own failures

**Where the failures come from.** By default the hard subset's ``healable`` conversations
(``data/golden/hub-v1.1/hard_subset.jsonl``): base Aya (Arm A) on the 112 train tasks the teacher
solved, outside validation, as the agent saw them. ``--episodes`` adds failed rollouts of another
model on those same tasks, e.g. SFT v2's (Charlie's ``S2-*-trainpool.jsonl``, SEL-73). Any other
task is skipped: validation, dev and test are never fixed.

**Points** (`reflexive.heal_data.points_of`): the first Tool, the first Arg and the first Loop
finding of each conversation (one point per turn). **Fixes**: `reflexive.heal.PrefixTeacher`
(GLM-5.3-Flash on OpenRouter, ``OPENROUTER_API_KEY``) sees the conversation up to that turn and
writes it again, up to 3 attempts, until `check_fix` accepts one: a real turn, no giving up, and no
Tool / Arg / Loop finding in it.

Writes ``data/heal/fixes.jsonl`` (accepted fixes, ``reagent.sft/v1``, labelled with the point's
types), ``data/heal/attempts.jsonl`` (every point tried: the attempts, refusals and usage) and
``data/heal/fixes.report.json``. A re-run skips the points already tried.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from reflexive import heal_data, splits
from reflexive.agent import Failure
from reflexive.files import ROOT, read_jsonl, shown
from reflexive.heal import PrefixTeacher

HARD = ROOT / "data" / "golden" / "hub-v1.1" / "hard_subset.jsonl"
OUT = ROOT / "data" / "heal"
GROUP = "healable"  # the train tasks the teacher solved, outside validation

Record = dict[str, Any]


def allowed_tasks(hard: list[Record]) -> set[tuple[str, str]]:
    return {(r["domain"], str(r["task_id"])) for r in hard if r.get("group") == GROUP}


def failed_records(hard: list[Record], episodes: list[Path], allowed: set[tuple[str, str]]) -> list[Record]:
    """The failed conversations to heal: the hard subset's (source ``armA``) and the episodes'."""
    out = []
    for r in hard:
        if r.get("group") == GROUP:
            out.append({**r, "point_source": "armA"})
    for p in episodes:
        name = p.name.removesuffix(".jsonl")
        for r in read_jsonl(p, partial_last_ok=True):
            task = (r["domain"], str(r["task_id"]))
            if task not in allowed or (r.get("outcome") or {}).get("passed") or r.get("error"):
                continue
            out.append({**r, "point_source": name})
    for r in out:
        splits.assert_not_test(r["domain"], str(r["task_id"]))
        if splits.split_of(r["domain"], str(r["task_id"])) != "train":
            raise SystemExit(f"⛔ {r['domain']}/{r['task_id']} isn't a train task: never fixed")
    return out


def point_id(r: Record, turn: int) -> str:
    sample = r.get("sample", 0)
    return f"{r['point_source']}/{r['domain']}/{r['task_id']}/s{sample}/t{turn}"


def points(records: list[Record]) -> list[tuple[str, Record, list[str], Any]]:
    """``[(point id, record, types, finding)]`` for every heal point."""
    out = []
    for r in records:
        for types, finding in heal_data.points_of(r):
            out.append((point_id(r, finding.turn), r, types, finding))
    return out


def fix_one(pid: str, rec: Record, types: list[str], finding: Any, teacher_kw: dict[str, Any]) -> Record:
    """Fix one point. Returns the attempts log, with ``example`` if a fix was accepted."""
    teacher = PrefixTeacher(**teacher_kw)
    failure = Failure(record=rec, diagnosis=heal_data.diagnosis_for(types, finding))
    example = teacher.fix(failure)
    log = {
        "point": pid,
        "types": types,
        "domain": rec["domain"],
        "task_id": str(rec["task_id"]),
        "source": rec["point_source"],
        "finding": finding.to_dict(),
        "accepted": example is not None,
        "attempts": teacher.tried.get(failure.task, []),
    }
    if example is not None:
        example["id"] = f"heal/{pid}"
        example["labels"] = list(types)
        example["source"]["point"] = pid
        example["source"]["point_source"] = rec["point_source"]
        log["example"] = example
    return log


def report(logs: list[Record]) -> Record:
    by = collections.defaultdict(lambda: [0, 0])
    for log in logs:
        for t in log["types"]:
            by[t][0] += 1
            by[t][1] += log["accepted"]
    usage = collections.Counter()
    for log in logs:
        for a in log["attempts"]:
            for k, v in (a.get("usage") or {}).items():
                if isinstance(v, int | float):
                    usage[k] += v
    refusals = collections.Counter(
        p.split(":")[0] for log in logs for a in log["attempts"] for p in a.get("problems") or []
    )
    return {
        "points": len(logs),
        "accepted": sum(log["accepted"] for log in logs),
        "by_type": {t: {"points": n, "accepted": a} for t, (n, a) in sorted(by.items())},
        "by_domain": dict(collections.Counter(log["domain"] for log in logs if log["accepted"])),
        "by_source": dict(collections.Counter(log["source"] for log in logs if log["accepted"])),
        "refusals": dict(refusals.most_common(10)),
        "usage": dict(usage),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hard", type=Path, default=HARD, help=f"default: {shown(HARD)}")
    p.add_argument("--episodes", nargs="*", default=[], help="more rollout files (globs) to heal")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--limit", type=int, help="only the first N new points (a smoke run)")
    p.add_argument("--workers", type=int, default=8, help="points fixed at once")
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--dry-run", action="store_true", help="count the points only: no API call")
    a = p.parse_args(argv)

    if not a.hard.exists():
        raise SystemExit(f"⛔ no {shown(a.hard)}: run scripts/pod/setup.sh (it downloads golden v1.1)")
    hard = read_jsonl(a.hard)
    episodes = sorted({Path(f) for pattern in a.episodes for f in glob.glob(pattern)})
    if a.episodes and not episodes:
        raise SystemExit(f"⛔ no files match {a.episodes}")
    pts = points(failed_records(hard, episodes, allowed_tasks(hard)))
    types = collections.Counter(t for _, _, ts, _ in pts for t in ts)
    convs = len({(r["domain"], r["task_id"], r["point_source"], r.get("sample", 0)) for _, r, _, _ in pts})
    print(
        f"{len(pts)} heal points from {convs} "
        f"failed conversations ({', '.join(f'{t} {types[t]}' for t in heal_data.TYPES)}); "
        f"sources: {dict(collections.Counter(r['point_source'] for _, r, _, _ in pts))}"
    )

    a.out.mkdir(parents=True, exist_ok=True)
    attempts_path, fixes_path = a.out / "attempts.jsonl", a.out / "fixes.jsonl"
    done = {log["point"] for log in read_jsonl(attempts_path)} if attempts_path.exists() else set()
    todo = [x for x in pts if x[0] not in done][: a.limit]
    print(f"{len(done)} already tried; {len(todo)} to fix now")
    if a.dry_run or not todo:
        return 0

    lock = threading.Lock()
    teacher_kw = {"attempts": a.attempts}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(fix_one, *x, teacher_kw) for x in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            log = fut.result()
            with lock:
                if example := log.pop("example", None):
                    with fixes_path.open("a") as f:
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                with attempts_path.open("a") as f:
                    f.write(json.dumps(log, ensure_ascii=False) + "\n")
            mark = "fixed" if example else "refused"
            print(f"[{i}/{len(todo)}] {mark} {log['point']} {'+'.join(log['types'])}", flush=True)

    rep = report(read_jsonl(attempts_path))
    (a.out / "fixes.report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(
        f"accepted {rep['accepted']}/{rep['points']}: "
        + ", ".join(f"{t} {v['accepted']}/{v['points']}" for t, v in rep["by_type"].items())
        + f"; written to {shown(fixes_path)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
