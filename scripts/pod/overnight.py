"""Run one arm to completion unattended, check that it really completed, then stop the pod.

    python scripts/pod/overnight.py S2E4 configs/sft/arms.s2e4.yaml --split test --stop
    python scripts/pod/overnight.py S2P1 configs/sft/arms.s2p1.yaml --split test --if-meets S2 --stop
    python scripts/pod/overnight.py S2E4 configs/sft/arms.s2e4.yaml --split test --check   # only the check

For each domain it runs `scripts/arms/rollout.py` on the split's tasks (``--samples`` each, the
guard off, written to ``results/episodes/arms/<arm>-<domain>-<split>.jsonl``), then checks the files:
**complete** means every task has ``--samples`` scored conversations (the last record per
conversation; crashed ones don't count). An incomplete run is retried, since `rollout.py` resumes and
re-runs only crashed or missing conversations, up to ``--retries`` times, ``--wait`` seconds apart
(rate limits or credit running out). Then it writes ``results/overnight-<arm>-<split>.json``.

``--stop`` stops this pod with ``runpodctl stop pod $RUNPOD_POD_ID`` at the end: after a complete
run, or after the last retry fails (the files are kept on the volume, and a re-run resumes), so an
idle pod isn't billed all night. Stopping keeps ``/workspace``. ``--if-meets BASELINE``: run only if
``results/stage2b-<BASELINE>-vs-<arm>-val.json`` says the arm meets Stage 2b's rule (otherwise just
stop, if ``--stop``). Validation is run by `val_rollout.py`, not here: this is for the test split.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from reflexive import splits
from reflexive.files import ROOT, latest_records, read_jsonl
from reflexive.metrics import is_broken

DOMAINS = ("retail", "airline", "telecom")
ARMS = ROOT / "results" / "episodes" / "arms"


def status(arm: str, split: str, samples: int, arms_dir: Path = ARMS) -> dict[str, dict]:
    """Per domain: tasks, scored conversations, crashed ones, and the tasks still short of ``samples``."""
    out = {}
    for d in DOMAINS:
        want = [str(t) for t in splits.task_ids(d, split)]
        f = arms_dir / f"{arm}-{d}-{split}.jsonl"
        recs = latest_records(read_jsonl(f, partial_last_ok=True)) if f.exists() else []
        scored = collections.Counter(str(r["task_id"]) for r in recs if not is_broken(r))
        crashed = sum(is_broken(r) for r in recs)
        short = [t for t in want if scored[t] < samples]
        out[d] = {"tasks": len(want), "scored": sum(scored.values()), "crashed": crashed, "short": short}
    return out


def complete(st: dict[str, dict]) -> bool:
    return all(not v["short"] for v in st.values())


def rollout(arm: str, domain: str, split: str, config: Path, samples: int, workers: int) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "arms" / "rollout.py"),
        arm,
        domain,
        "--split",
        split,
        "--samples",
        str(samples),
        "--config",
        str(config),
        "--out",
        str(ARMS / f"{arm}-{domain}-{split}.jsonl"),
        "--actions",
        "off",
        "--workers",
        str(workers),
    ]
    return subprocess.run(cmd, check=False).returncode


def stop_pod(dry: bool = False) -> str:
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        return "not stopped: no RUNPOD_POD_ID (not on a RunPod pod?)"
    if dry:
        return f"would stop pod {pod}"
    r = subprocess.run(["runpodctl", "stop", "pod", pod], capture_output=True, text=True, check=False)
    return f"runpodctl stop pod {pod}: exit {r.returncode} {r.stdout.strip()} {r.stderr.strip()}"


def meets(baseline: str, arm: str) -> bool | None:
    f = ROOT / "results" / f"stage2b-{baseline}-vs-{arm}-val.json"
    return json.loads(f.read_text()).get("meets_rule") if f.exists() else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm")
    p.add_argument("config", type=Path)
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--samples", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=6)
    p.add_argument("--wait", type=int, default=600, help="seconds between retries")
    p.add_argument("--if-meets", metavar="BASELINE")
    p.add_argument("--stop", action="store_true", help="stop this pod at the end")
    p.add_argument("--check", action="store_true", help="only report how complete the files are")
    a = p.parse_args()
    report_path = ROOT / "results" / f"overnight-{a.arm}-{a.split}.json"

    def finish(outcome: str, st: dict) -> int:
        report = {"arm": a.arm, "split": a.split, "samples": a.samples, "outcome": outcome, "status": st}
        report["finished"] = dt.datetime.now().isoformat(timespec="seconds")
        if a.stop:
            report["stop"] = stop_pod()
        report_path.write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report, indent=1))
        return 0 if outcome == "complete" else 1

    st = status(a.arm, a.split, a.samples)
    if a.check:
        print(json.dumps({d: {**v, "short": len(v["short"])} for d, v in st.items()}, indent=1))
        return 0 if complete(st) else 1
    if a.if_meets is not None:
        verdict = meets(a.if_meets, a.arm)
        print(f"{a.arm} against {a.if_meets} on validation: meets the rule = {verdict}")
        if verdict is not True:
            return finish("skipped: doesn't meet Stage 2b's rule on validation", st)
    for attempt in range(a.retries + 1):
        if attempt:
            short = {d: len(v["short"]) for d, v in st.items()}
            print(f"incomplete ({short} tasks short): retry {attempt} in {a.wait}s")
            time.sleep(a.wait)
        for d in DOMAINS:
            if st[d]["short"]:
                rollout(a.arm, d, a.split, a.config, a.samples, a.workers)
        st = status(a.arm, a.split, a.samples)
        if complete(st):
            return finish("complete", st)
    return finish(f"incomplete after {a.retries} retries", st)


if __name__ == "__main__":
    raise SystemExit(main())
