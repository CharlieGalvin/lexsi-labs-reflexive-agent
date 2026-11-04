"""Run one experiment arm (configs/arms.yaml) on τ²-bench tasks and save the trajectories (SEL-41).

    python scripts/rollout.py A retail --split train --limit 5
    python scripts/rollout.py C telecom --task-ids <id> <id> --samples 3

One JSON line per conversation in ``results/episodes/arms/<arm>-<domain>-<split>.jsonl``, in the
`Tau2Episode.record()` format plus ``arm``, ``sample`` and diagnostics. An interrupted run
resumes: (task, sample) pairs already in the file are skipped. ``--fresh`` starts again.

**Crashes** (rate limits, server errors: records with ``error``) aren't model failures. The
summary counts them apart, and a re-run retries them (their records are replaced).
**Rate limits**: ``--rpm`` (or ``rpm`` in the config) caps the student's calls per minute across
all workers. A Cohere Trial key allows 20/min and 1,000 calls a month.

**Splits** are τ²-bench's own train / test lists until the frozen dev split (SEL-36) is merged.
**Judges**: retail and airline need ``OPENAI_API_KEY`` for τ²-bench's LLM judges. Without it
(``--judges auto``) only the deterministic checks are scored, and the record lists what was
skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from reflexive.arms import get_arm, make_student, run_episode
from reflexive.episode import _split
from reflexive.teacher import RateLimiter


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", help="arm key from configs/arms.yaml, e.g. A or C")
    ap.add_argument("domain", choices=["retail", "airline", "telecom"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--task-ids", nargs="+", help="these tasks instead of the whole split")
    ap.add_argument("--limit", type=int, help="only the first N tasks of the split (sorted)")
    ap.add_argument("--samples", type=int, default=1, help="conversations per task")
    ap.add_argument("--workers", type=int, default=4, help="conversations run at once")
    ap.add_argument("--judges", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--config", type=pathlib.Path, help="arms config (default: configs/arms.yaml)")
    ap.add_argument("--out", type=pathlib.Path, help="output .jsonl")
    ap.add_argument("--rpm", type=float, help="student calls/min for the whole run (default: config)")
    ap.add_argument("--fresh", action="store_true", help="overwrite earlier results")
    a = ap.parse_args()

    arm = get_arm(a.arm, a.config)
    judges = {"on": True, "off": False}.get(a.judges, bool(os.environ.get("OPENAI_API_KEY")))
    tasks = a.task_ids or sorted(_split(a.domain, a.split))
    if a.task_ids:
        split_of = {t: s for s in ("train", "test") for t in _split(a.domain, s)}
        a.split = "-".join(sorted({split_of.get(t, "other") for t in tasks}))
    if a.limit:
        tasks = tasks[: a.limit]
    out = a.out or pathlib.Path(f"results/episodes/arms/{arm.key}-{a.domain}-{a.split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []  # clean records from earlier runs; crashed ones are dropped and re-run
    if out.exists() and not a.fresh:
        kept = [line for line in out.read_text().splitlines()
                if line.strip() and not json.loads(line).get("error")]
    done = {(r["task_id"], r.get("sample", 0)) for r in map(json.loads, kept)}
    todo = [(t, s) for t in tasks for s in range(a.samples) if (t, s) not in done]
    rpm = a.rpm if a.rpm is not None else arm.student.get("rpm") or 0
    print(f"Arm {arm.key} ({arm.name}), {arm.student['name']}, {a.domain} {a.split}: "
          f"{len(todo)} conversations to run ({len(done)} already in {out}); judges "
          f"{'on' if judges else 'off'}; student rpm {rpm or 'unlimited'}")
    if not todo:
        return 0

    try:
        make_student(arm.student)  # fail fast on a missing API key, before any conversation
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    limiter = RateLimiter(rpm)  # one for the whole run, shared by every worker
    lock = threading.Lock()
    passed = errors = overflows = 0

    def one(task_id: str, sample: int) -> dict:
        return run_episode(arm, make_student(arm.student, limiter=limiter), a.domain, task_id,
                           judges=judges, sample=sample)

    if out.exists() and not a.fresh:  # rewrite without the crashed records being re-run
        out.write_text("".join(f"{line}\n" for line in kept))
    with out.open("w" if a.fresh else "a") as f, ThreadPoolExecutor(a.workers) as pool:
        futures = {pool.submit(one, t, s): (t, s) for t, s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            errors += bool(rec.get("error"))
            passed += rec["outcome"]["passed"] and not rec.get("error")
            overflows += rec["flags"].get("context_overflow", False)
            t, s = futures[fut]
            mark = "PASS" if rec["outcome"]["passed"] else "fail"
            print(f"[{i}/{len(todo)}] {mark} {t} #{s} ({rec['termination']}, {rec['wall_s']}s)"
                  + (f" ERROR {rec['error']}" if rec.get("error") else ""))

    print(f"\nThis run: {passed}/{len(todo) - errors} passed, {overflows} context overflows, "
          f"{errors} crashed (not scored; a re-run retries them). Saved to {out}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
