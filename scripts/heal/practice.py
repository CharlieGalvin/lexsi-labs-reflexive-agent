"""The self-practice loop's round (Stage 2b, STAGE2.md; SEL-23): practise, verify, build the data.

    python scripts/heal/practice.py run --from $VALDIR                 # S2PR on the train pool, 2 samples
    python scripts/heal/practice.py run --from $VALDIR --dry-run       # the task lists and commands only
    python scripts/heal/practice.py build                              # → data/train/sft-p1.jsonl
    python scripts/heal/practice.py build --no-judge                   # skip the "no yes" filter (a check)

**run**: the practising arm (default S2PR, `configs/sft/arms.s2pr.yaml`: v2 at temperature 0.7)
on every train task outside validation, ``--samples`` attempts each, written to
``results/episodes/arms/<arm>-<domain>-trainpool.jsonl``. Validation tasks are read from
``P-<domain>-val.jsonl`` in ``--from`` and left out; anything outside the train split is refused.
A re-run resumes (`rollout.py`). Anything else on the command line goes to `rollout.py`.

**build**: the round's training file. A practice conversation is kept only if all hold:

- τ²-bench passed it and it isn't broken (`reflexive.metrics.is_broken`);
- it is clean (`golden.why_not_clean`: no tool errors, the agent's own view);
- the frozen checker's "no yes" rule finds no change made without the customer's yes (the
  training-data audit showed "mismatch" is unreliable on conversations that passed, so it isn't
  used as a filter);
- it is a train task outside validation (checked again here).

At most ``--per-task`` (default 2) per task. Each is formatted as v2's examples are (Cohere format,
every agent turn trained on), group ``practice``, and appended to v2's file (``--base``); tasks the
model never passed keep only the teacher's conversation already in v2's data. Writes
``--out`` and ``<out>.report.json`` (counts per domain and why conversations were left out).
"""

from __future__ import annotations

import argparse
import collections
import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from reflexive import splits
from reflexive.files import ROOT, latest_records, read_jsonl

DOMAINS = ("retail", "airline", "telecom")
ARMS_DIR = ROOT / "results" / "episodes" / "arms"
BASE = ROOT / "data" / "train" / "sft-v2.jsonl"
OUT = ROOT / "data" / "train" / "sft-p1.jsonl"
CONFIG = ROOT / "configs" / "sft" / "arms.s2pr.yaml"
CACHE = ROOT / "results" / "cache"

Record = dict[str, Any]
Flags = Callable[[Record], list[dict]]


def _val_rollout() -> Any:
    spec = importlib.util.spec_from_file_location("val_rollout", ROOT / "scripts" / "arms" / "val_rollout.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pool(domain: str, val: set[str]) -> list[str]:
    """The train tasks of ``domain`` outside validation, sorted as strings."""
    return sorted(str(t) for t in splits.task_ids(domain, "train") if str(t) not in val)


def run(a: argparse.Namespace, extra: list[str]) -> int:
    vr = _val_rollout()
    for d in a.domains:
        val = set(vr.val_tasks(a.source, d))
        ids = pool(d, val)
        out = ARMS_DIR / f"{a.arm}-{d}-trainpool.jsonl"
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "arms" / "rollout.py"),
            a.arm,
            d,
            "--task-ids",
            *ids,
            "--samples",
            str(a.samples),
            "--config",
            str(a.config),
            "--out",
            str(out),
            "--actions",
            "off",
            *extra,
        ]
        print(
            f"{a.arm} practising {d}: {len(ids)} train tasks (validation's {len(val)} left out) × {a.samples}"
        )
        if a.dry_run:
            print("   " + " ".join(cmd[1:6]) + " … " + " ".join(cmd[-8:]))
            continue
        if (code := subprocess.run(cmd, check=False).returncode) != 0:
            print(f"⛔ {d} stopped (exit {code}): re-run the same command to resume")
            return code
    if not a.dry_run:
        print("done: python scripts/heal/practice.py build")
    return 0


def no_yes_flags(rec: Record) -> list[dict]:
    """The frozen "no yes" rule's live findings on every change in ``rec`` (as the guard checks)."""
    from reflexive.intent import intent_findings, no_yes

    return [f for f in intent_findings(rec, rules=[no_yes]) if f.get("live")]


def to_example(rec: Record, k: int, arm: str) -> Record:
    """One practice conversation as a training example, formatted as v2's are."""
    from reflexive import golden
    from reflexive.hallucination import agent_view_of, domain_tools
    from reflexive.heal import SFT_FORMAT
    from reflexive.teacher import to_prompted

    tools, _ = domain_tools(rec["domain"])
    system_prompt = agent_view_of(rec)[0]["content"]
    view = golden.agent_view(rec["messages"], system_prompt)
    msgs = [{"role": m["role"], "content": m["content"]} for m in to_prompted(view, tools, "cohere")]
    return {
        "id": f"practice/{arm}/{rec['domain']}/{rec['task_id']}/s{rec.get('sample', 0)}",
        "format": SFT_FORMAT,
        "domain": rec["domain"],
        "task_id": str(rec["task_id"]),
        "group": "practice",
        "labels": ["stage2b-practice-r1"],
        "messages": msgs,
        "train_on": [i for i, m in enumerate(msgs) if m["role"] == "assistant"],
        "source": {
            "kind": "self_practice",
            "arm": arm,
            "sample": rec.get("sample", 0),
            "student": rec.get("agent_llm"),
            "user_llm": rec.get("user_llm"),
            "outcome": rec.get("outcome"),
            "kept_rank": k,
        },
    }


def select(
    recs: Iterable[Record], val: set[tuple[str, str]], per_task: int, flags: Flags | None
) -> tuple[list[Record], collections.Counter, dict[str, set[str]]]:
    """The practice conversations to keep, why the rest were left out, and the tasks passed per domain."""
    from reflexive import golden
    from reflexive.metrics import is_broken

    keep: list[Record] = []
    why: collections.Counter = collections.Counter()
    count: collections.Counter = collections.Counter()
    passed: dict[str, set[str]] = collections.defaultdict(set)
    for r in sorted(latest_records(recs), key=lambda r: (r["domain"], str(r["task_id"]), r.get("sample", 0))):
        d, t = r["domain"], str(r["task_id"])
        splits.assert_not_test(d, t)
        if splits.split_of(d, t) != "train" or (d, t) in val:
            raise SystemExit(f"⛔ {d}/{t} isn't a train task outside validation: practice can't use it")
        if is_broken(r):
            why["broken (crash, overflow)"] += 1
        elif not (r.get("outcome") or {}).get("passed"):
            why["failed"] += 1
        elif (bad := golden.why_not_clean(r)) is not None:
            why[f"not clean: {bad}"] += 1
        elif flags is not None and flags(r):
            why['"no yes" flagged a change'] += 1
        elif count[(d, t)] >= per_task:
            why[f"over {per_task} per task"] += 1
        else:
            count[(d, t)] += 1
            passed[d].add(t)
            keep.append(r)
    return keep, why, passed


def build(a: argparse.Namespace) -> int:
    vr = _val_rollout()
    val = {(d, t) for d in DOMAINS for t in vr.val_tasks(a.source, d)}
    recs: list[Record] = []
    for d in DOMAINS:
        f = ARMS_DIR / f"{a.arm}-{d}-trainpool.jsonl"
        if not f.exists():
            raise SystemExit(f"⛔ no {f.relative_to(ROOT)}: run `practice.py run` first")
        recs += read_jsonl(f, partial_last_ok=True)
    flags: Flags | None = None
    if not a.no_judge:
        os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
        os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
        flags = no_yes_flags
    keep, why, passed = select(recs, val, a.per_task, flags)
    base = read_jsonl(a.base)
    if any(r.get("group") == "practice" for r in base):
        raise SystemExit(f"⛔ {a.base} already has practice examples: build from v2's file")
    exs = [to_example(r, i, a.arm) for i, r in enumerate(keep)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r) + "\n" for r in [*base, *exs]))
    taught = {(r["domain"], str(r["task_id"])) for r in base}
    report = {
        "practice_conversations": len(latest_records(recs)),
        "kept": len(exs),
        "left_out": dict(why),
        "per_domain": {
            d: {"kept": sum(e["domain"] == d for e in exs), "tasks_passed": len(passed.get(d, set()))}
            for d in DOMAINS
        },
        "tasks_with_practice": len({(e["domain"], e["task_id"]) for e in exs}),
        "practice_tasks_not_in_v2_data": sorted(
            f"{e['domain']}/{e['task_id']}" for e in exs if (e["domain"], e["task_id"]) not in taught
        ),
        "base_examples": len(base),
        "out": str(a.out),
        "no_yes_filter": not a.no_judge,
    }
    Path(f"{a.out}.report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "practice_tasks_not_in_v2_data"}, indent=1))
    print(f"→ {a.out}: {len(base)} of v2's + {len(exs)} practice")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "build"):
        s = sub.add_parser(name)
        s.add_argument("--arm", default="S2PR")
        s.add_argument(
            "--from", dest="source", type=Path, default=ARMS_DIR, help="where P-<domain>-val.jsonl are"
        )
    r = sub.choices["run"]
    r.add_argument("--config", type=Path, default=CONFIG)
    r.add_argument("--samples", type=int, default=2)
    r.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    r.add_argument("--dry-run", action="store_true")
    b = sub.choices["build"]
    b.add_argument("--base", type=Path, default=BASE)
    b.add_argument("--out", type=Path, default=OUT)
    b.add_argument("--per-task", type=int, default=2)
    b.add_argument(
        "--no-judge", action="store_true", help='skip the "no yes" filter (a check, not the round)'
    )
    a, extra = p.parse_known_args(argv)
    if a.cmd == "run":
        return run(a, extra)
    if extra:
        p.error(f"unrecognised arguments: {' '.join(extra)}")
    return build(a)


if __name__ == "__main__":
    raise SystemExit(main())
