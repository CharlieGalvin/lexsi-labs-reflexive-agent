"""Stage 2 training data (SEL-102): v2's data + the mismatch fixes + the guard's fixes on a go.

    python scripts/data/build_stage2_data.py --decision1 go     # + guard fixes from S2G-*-trainpool.jsonl
    python scripts/data/build_stage2_data.py --decision1 no-go  # mismatch fixes only
    python scripts/data/build_stage2_data.py --decision1 go --dry-run

As pre-registered in STAGE2.md ("Retraining", amended Sun 22 Nov 10:40):

* **v2's data** (``data/train/sft-v2.jsonl``) unchanged: groups ``healable`` and ``extra``.
* **Mismatch fixes** (group ``mismatch``, ``results/heal/mismatch-fixes.jsonl`` from
  ``onpolicy_fixes.py --points mismatch``): every one, as accepted.
* **Guard fixes** (group ``guard``), only with ``--decision1 go``: v2 + guard's conversations on the
  train-pool tasks (``S2G-<domain>-trainpool.jsonl``, ``--actions act --rules mismatch``) that
  τ²-bench passes and that are clean (`golden.why_not_clean`). Each is converted exactly as v2's
  extra passes were (`build_extra_fixes.to_example`). A blocked reply and its re-ask never reach the
  saved conversation (the guard only adds them to the agent's prompt), so nothing of the guard is
  trained on.
* **Too few:** if the fixes used (mismatch, plus guard on a go) number fewer than 30, nothing is
  written: report the count and skip retraining (v2 + guard goes to Decision 2).

Every task is checked: a train task, never a validation task (``P-*-val.jsonl``, which must be
whole, as for the fixes), never test (`assert_not_test`). A word-for-word repeat of an example
already kept is left out. Writes ``data/train/sft-s2.jsonl`` and a report beside it.
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from reflexive import golden
from reflexive.files import ROOT, latest_records, read_jsonl, sha256, shown
from reflexive.splits import assert_not_test, split_of

BASE = ROOT / "data" / "train" / "sft-v2.jsonl"
MISMATCH = ROOT / "results" / "heal" / "mismatch-fixes.jsonl"
EPISODES = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "data" / "train" / "sft-s2.jsonl"
MIN_FIXES = 30
DOMAINS = ("retail", "airline", "telecom")
Record = dict[str, Any]


def _script(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


class TooFew(Exception):
    pass


def check_task(domain: str, task_id: str, banned: set[tuple[str, str]]) -> str | None:
    """Why a task can't be trained on (None if it can)."""
    assert_not_test(domain, task_id)
    if (domain, task_id) in banned:
        return "a validation task"
    if split_of(domain, task_id) != "train":
        return "not a train task"
    return None


def select_guard(
    recs: Iterable[Record], banned: set[tuple[str, str]]
) -> tuple[list[Record], collections.Counter]:
    """The guarded conversations that can become guard fixes, and why the rest can't."""
    keep, why = [], collections.Counter()
    for r in recs:
        d, t = r["domain"], str(r["task_id"])
        if (bad := check_task(d, t, banned)) is not None:
            why[bad] += 1
        elif (bad := golden.why_not_clean(r)) is not None:
            why[bad] += 1
        else:
            keep.append(r)
    return keep, why


def guard_blocks(r: Record) -> int:
    """How many replies the guard blocked in a conversation (``rec["monitor"]["actions"]``)."""
    acts = ((r.get("monitor") or {}).get("actions") or {}).get("blocks") or []
    return sum(1 for e in acts if e.get("action") == "blocked")


def check_mismatch(exs: Iterable[Record], banned: set[tuple[str, str]]) -> list[str]:
    """Everything wrong with the mismatch fixes as training examples (empty if nothing is)."""
    out = []
    for ex in exs:
        rid = ex.get("id", "?")
        if ex.get("group") != "mismatch":
            out.append(f"{rid}: group {ex.get('group')!r}, expected 'mismatch'")
        if ex.get("format") != "reagent.sft/v1":
            out.append(f"{rid}: format {ex.get('format')!r}")
        if (bad := check_task(ex["domain"], str(ex["task_id"]), banned)) is not None:
            out.append(f"{rid}: {bad}")
        n = len(ex.get("messages") or [])
        if not ex.get("train_on") or any(not 0 <= i < n for i in ex["train_on"]):
            out.append(f"{rid}: train_on doesn't point into its messages")
    return out


def build(
    base: list[Record],
    mismatch: list[Record],
    guard: list[Record],
    decision1: str,
    to_example: Callable[[Record], Record],
) -> tuple[list[Record], dict]:
    """v2's examples + the fixes, deduplicated; raises `TooFew` under the pre-registered 30."""
    seen = {json.dumps([m["content"] for m in r["messages"]]) for r in base}
    added: list[Record] = []
    repeats = collections.Counter()
    candidates = [("mismatch", ex) for ex in mismatch]
    if decision1 == "go":
        for r in guard:
            ex = to_example(r)
            ex.update(group="guard", labels=["stage2-guard"])
            ex["source"] = {**ex.get("source", {}), "kind": "guard_fix", "guard_blocks": guard_blocks(r)}
            candidates.append(("guard", ex))
    for group, ex in candidates:
        text = json.dumps([m["content"] for m in ex["messages"]])
        if text in seen:
            repeats[group] += 1
            continue
        seen.add(text)
        added.append(ex)
    per_group = collections.Counter(ex["group"] for ex in added)
    used = per_group["mismatch"] + per_group["guard"]
    counts = {
        "decision1": decision1,
        "base": len(base),
        "fixes_used": used,
        "per_group": dict(per_group),
        "per_domain": dict(collections.Counter(ex["domain"] for ex in added)),
        "distinct_tasks": len({(ex["domain"], str(ex["task_id"])) for ex in added}),
        "guard_fixes_with_a_block": sum(1 for ex in added if (ex.get("source") or {}).get("guard_blocks")),
        "repeats_left_out": dict(repeats),
    }
    if used < MIN_FIXES:
        raise TooFew(f"{used} fixes (< {MIN_FIXES}): skip retraining, v2 + guard goes to Decision 2")
    return [*base, *added], counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decision1", required=True, choices=["go", "no-go"], help="Decision 1's result (SEL-96)")
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--mismatch", type=Path, default=MISMATCH)
    p.add_argument("--episodes", type=Path, default=EPISODES, help="where S2G-<domain>-trainpool.jsonl are")
    p.add_argument("--exclude", nargs="*", default=sorted(glob.glob(str(EPISODES / "P-*-val.jsonl"))))
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true", help="count only, write nothing")
    a = p.parse_args(argv)

    from loguru import logger

    logger.remove()
    onpolicy = _script("onpolicy_fixes", "scripts/heal/onpolicy_fixes.py")
    banned, problems = onpolicy.val_ids(a.exclude)
    if problems:
        print("⛔ the validation exclusion isn't complete, refusing to risk training on validation tasks:")
        for why in problems:
            print(f"   {why}")
        return 1
    base = read_jsonl(a.base)
    mismatch = read_jsonl(a.mismatch)
    if bad := check_mismatch(mismatch, banned):
        for line in bad:
            print(f"⛔ {line}")
        return 1
    guard, why = [], collections.Counter()
    if a.decision1 == "go":
        recs = []
        for d in DOMAINS:
            f = a.episodes / f"S2G-{d}-trainpool.jsonl"
            if not f.exists():
                print(f"⛔ {shown(f)} not found: the guard-fix run (SEL-100) must finish first")
                return 1
            recs += latest_records(read_jsonl(f, partial_last_ok=True))
        guard, why = select_guard(recs, banned)
    extra = _script("build_extra_fixes", "scripts/data/build_extra_fixes.py")
    golden_mod = _script("build_golden", "scripts/data/build_golden.py")
    systems: dict[str, Any] = {}

    def to_example(r: Record) -> Record:
        d = r["domain"]
        if d not in systems:
            systems[d] = golden_mod.prompts_and_tools(d, str(r["task_id"]))
        return extra.to_example(r, systems[d])

    try:
        rows, counts = build(base, mismatch, guard, a.decision1, to_example)
    except TooFew as e:
        print(f"⛔ {e}")
        return 1
    counts["guard_left_out"] = dict(why)
    print(f"Stage 2 data: {counts['base']} v2 examples + {counts['fixes_used']} fixes {counts['per_group']}")
    print(f"  {counts['distinct_tasks']} distinct tasks; per domain {counts['per_domain']}")
    if a.decision1 == "go":
        print(
            f"  guard fixes with a block in them: {counts['guard_fixes_with_a_block']}; left out: {dict(why)}"
        )
    if counts["repeats_left_out"]:
        print(f"  word-for-word repeats left out: {counts['repeats_left_out']}")
    if a.dry_run:
        print("(dry run: nothing written)")
        return 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {
        "sel": "SEL-102",
        "base": {"file": shown(a.base), "sha256": sha256(a.base)},
        "mismatch": {"file": shown(a.mismatch), "sha256": sha256(a.mismatch)},
        **counts,
    }
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"→ {shown(a.out)} ({len(rows)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
