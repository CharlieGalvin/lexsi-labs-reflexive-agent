"""How often the checker's rules flag train-task conversations, and whether they catch injected errors.

    just rule-rates                                    # SFT v2 (S2) on the train pool, both rules
    just rule-rates --arms S2:trainpool,A:train        # + base Aya (Arm A) on the train tasks
    just rule-rates --inject 40                        # + 40 passing changes corrupted on purpose
    just rule-rates --show 5                           # print 5 flagged passes, to check by hand

Train tasks only. Every conversation goes through `assert_not_test`, and the validation tasks
(read from ``P-*-val.jsonl``, as `scripts/arms/val_rollout.py` does) are dropped, because Decision 1
is scored on them and nobody may look first. An arm other than S2's train pool needs those files.

For each rule ("no yes", "mismatch", and either): the share of failures flagged, of the failures
no Tool / Arg / Loop finding explains, and of passes (a flagged pass is a change the guard would
have blocked in a conversation that went well: STAGE2.md's bar is 1 in 10).

``--inject N``: N changes from passing conversations, each corrupted once (an item swapped for
another item ID the agent saw, an item dropped or added, another order, a flight dropped, a date
moved, one more bag). The mismatch rule should flag the corrupted call and not the original.
Injected errors are easier than real ones: a sanity check of the rule, not a headline.

Needs OPENAI_API_KEY; every answer is cached (results/cache/), so a re-run is free.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reflexive.hallucination import FailureType, agent_view_of, classify_record
from reflexive.metrics import is_broken
from reflexive.splits import assert_not_test, split_of
from reflexive.stats import wilson

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
CACHE = ROOT / "results" / "cache"
DOMAINS = ("retail", "airline", "telecom")


def val_tasks(episodes: Path) -> set[tuple[str, str]] | None:
    out: set[tuple[str, str]] = set()
    found = False
    for d in DOMAINS:
        f = episodes / f"P-{d}-val.jsonl"
        if f.exists():
            found = True
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        out.add((d, str(json.loads(line)["task_id"])))
                    except (json.JSONDecodeError, KeyError):
                        continue
    return out if found else None


def load(episodes: Path, arm: str, split: str, val: set[tuple[str, str]]) -> tuple[list[dict], int]:
    recs: dict[tuple, dict] = {}
    dropped = 0
    for d in DOMAINS:
        f = episodes / f"{arm}-{d}-{split}.jsonl"
        if not f.exists():
            print(f"⚠ {f.name} not found: left out")
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            if not r or is_broken(r):
                continue
            t = str(r["task_id"])
            assert_not_test(d, t)
            if (d, t) in val or split_of(d, t) != "train":
                dropped += 1
                continue
            recs[(arm, d, t, r.get("sample", 0))] = {**r, "_arm": arm}
    return list(recs.values()), dropped


def share(k: int, n: int) -> str:
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    return f"{k:>3}/{n:<3} {(k / n if n else 0):>4.0%} [{lo:.0%}–{hi:.0%}]"


# --- Injected errors --------------------------------------------------------------------------

ITEM_ID = re.compile(r'"item_id":\s*"(\d{6,})"')
ORDER_ID = re.compile(r"#W\d{7}")


def _args(call: dict) -> dict:
    a = (call.get("function") or call).get("arguments")
    try:
        return json.loads(a) if isinstance(a, str) else dict(a or {})
    except (json.JSONDecodeError, TypeError, ValueError):
        return {}


def _with(call: dict, args: dict) -> dict:
    c = copy.deepcopy(call)
    c["function"]["arguments"] = json.dumps(args)
    return c


def corrupt(call: dict, seen: str, rng: random.Random) -> tuple[dict, str] | None:
    """One wrong version of ``call``, made from what the agent saw; None if there's no clean way."""
    name = (call.get("function") or call).get("name")
    a = _args(call)
    items = [i for i in ITEM_ID.findall(seen)]
    if name in ("modify_pending_order_items", "exchange_delivered_order_items") and a.get("new_item_ids"):
        others = sorted(set(items) - set(a["new_item_ids"]) - set(a.get("item_ids") or []))
        if others:
            new = list(a["new_item_ids"])
            k = rng.randrange(len(new))
            new[k] = rng.choice(others)
            return _with(call, {**a, "new_item_ids": new}), "a new item swapped for another item ID it saw"
    if name == "return_delivered_order_items" and a.get("item_ids"):
        ids = list(a["item_ids"])
        if len(ids) > 1:
            return _with(call, {**a, "item_ids": ids[:-1]}), "an item dropped from the return"
        others = sorted(set(items) - set(ids))
        if others:
            return _with(call, {**a, "item_ids": [*ids, rng.choice(others)]}), "an extra item returned"
    if name == "cancel_pending_order" and a.get("order_id"):
        others = sorted(set(ORDER_ID.findall(seen)) - {a["order_id"]})
        if others:
            return _with(call, {**a, "order_id": rng.choice(others)}), "another order cancelled"
    if name == "update_reservation_flights" and a.get("flights"):
        fl = list(a["flights"])
        if len(fl) > 1:
            return _with(call, {**a, "flights": fl[:-1]}), "a flight dropped"
        f0 = dict(fl[0])
        m = re.match(r"(\d{4}-\d{2})-(\d{2})$", str(f0.get("date", "")))
        if m and int(m.group(2)) < 28:
            f0["date"] = f"{m.group(1)}-{int(m.group(2)) + 1:02d}"
            return _with(call, {**a, "flights": [f0]}), "a flight's date moved a day"
    if name == "update_reservation_baggages" and isinstance(a.get("total_baggages"), int):
        return _with(
            call,
            {
                **a,
                "total_baggages": a["total_baggages"] + 1,
                "nonfree_baggages": int(a.get("nonfree_baggages") or 0) + 1,
            },
        ), "one more bag than asked"
    return None


def injected(recs: list[dict], n: int, rule: Callable[..., Any], seed: int = 0) -> dict:
    from reflexive.mismatch import TOOLS

    rng = random.Random(seed)
    points = []
    for r in sorted(
        (r for r in recs if r["outcome"]["passed"]), key=lambda r: (r["_arm"], r["domain"], str(r["task_id"]))
    ):
        view = agent_view_of(r)
        for i, m in enumerate(view):
            for c in (m.get("tool_calls") or []) if m.get("role") == "assistant" else []:
                if (c.get("function") or {}).get("name") in TOOLS:
                    points.append((r, view, i, c))
    rng.shuffle(points)
    out = {"tried": 0, "caught": 0, "clean_flagged": 0, "kinds": {}}
    for _r, view, i, c in points:
        if out["tried"] >= n:
            break
        seen = "\n".join(str(x.get("content")) for x in view[:i] if x.get("role") == "tool")
        bad = corrupt(c, seen, rng)
        if bad is None:
            continue
        call, kind = bad
        policy = str(view[0].get("content")) if view and view[0].get("role") == "system" else ""
        out["tried"] += 1
        hit = rule(view[:i], call, policy)
        clean = rule(view[:i], c, policy)
        for f in (hit, clean):
            if f is not None and not f.live:
                raise RuntimeError(f"checker error: {f.reason}")
        k = out["kinds"].setdefault(kind, [0, 0])
        k[0] += hit is not None
        k[1] += 1
        out["caught"] += hit is not None
        out["clean_flagged"] += clean is not None
    return out


# --- Main -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one report, read top to bottom
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--arms", default="S2:trainpool", help="ARM:SPLIT pairs, comma-separated")
    p.add_argument("--inject", type=int, default=0, help="this many corrupted changes, for the mismatch rule")
    p.add_argument("--show", type=int, default=0, help="print this many flagged passes")
    p.add_argument("--json", type=Path, help="also write the numbers here")
    a = p.parse_args(argv)
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    os.environ.setdefault("REFLEXIVE_META_CACHE", str(CACHE / "meta-rules.jsonl"))

    from reflexive.intent import intent_findings, no_yes
    from reflexive.mismatch import mismatch

    pairs = [tuple(x.split(":", 1)) for x in a.arms.split(",") if x]
    val = val_tasks(a.episodes)
    if val is None and pairs != [("S2", "trainpool")]:
        sys.exit(
            "⛔ no P-*-val.jsonl found: can't drop the validation tasks, so only S2:trainpool is allowed"
        )
    recs: list[dict] = []
    for arm, split in pairs:
        rs, dropped = load(a.episodes, arm, split, val or set())
        print(
            f"{arm}:{split}: {len(rs)} conversations"
            + (f" ({dropped} validation/non-train dropped)" if dropped else "")
        )
        recs += rs
    if not recs:
        sys.exit("no conversations found")

    out: dict = {"arms": a.arms, "rules": {}}
    rows: dict = {}
    errors, shown = [], 0
    for r in recs:
        found = {
            "no_yes": intent_findings(r, rules=[no_yes]),
            "mismatch": intent_findings(r, rules=[mismatch]),
        }
        for fs in found.values():
            errors += [f["reason"] for f in fs if not f["live"]]
        live = {k: [f for f in fs if f["live"]] for k, fs in found.items()}
        live["either"] = live["no_yes"] + live["mismatch"]
        passed = r["outcome"]["passed"]
        unexplained = not passed and classify_record(r).failure == FailureType.NONE
        for rule, fs in live.items():
            for key in ("failures", "unexplained failures", "passes"):
                if (key == "passes") != passed or (key == "unexplained failures" and not unexplained):
                    continue
                row = rows.setdefault((r["_arm"], r["domain"], rule, key), [0, 0])
                row[0] += bool(fs)
                row[1] += 1
        if passed and live["mismatch"] and shown < a.show:
            shown += 1
            print(
                f"\n--- {r['_arm']} {r['domain']} task {r['task_id']} #{r.get('sample', 0)}: "
                "passed, mismatch flagged"
            )
            for f in live["mismatch"]:
                print(f"    turn {f['turn']} {f['tool']}: {f['reason'][:400]}")
    if errors:
        print(f"\n⚠ {len(errors)} checker errors: those changes weren't flagged, so the rates are too low.")
        print(f"  first: {errors[0]}\n  Fix it (usually OPENAI_API_KEY) and run again; nothing is written.")
        return 1

    for arm, _ in pairs:
        print(f"\n== {arm}")
        for rule in ("no_yes", "mismatch", "either"):
            print(f"  {rule}")
            for d in DOMAINS:
                cells = [
                    share(*rows.get((arm, d, rule, k), [0, 0]))
                    for k in ("failures", "unexplained failures", "passes")
                ]
                if any(rows.get((arm, d, rule, k)) for k in ("failures", "passes")):
                    print(f"    {d:8s} failures {cells[0]}  unexplained {cells[1]}  passes {cells[2]}")
                    out["rules"].setdefault(f"{arm}/{rule}", {})[d] = {
                        k: rows.get((arm, d, rule, k), [0, 0])
                        for k in ("failures", "unexplained failures", "passes")
                    }
    if a.inject:
        inj = injected(recs, a.inject, mismatch)
        out["injected"] = inj
        print(
            f"\ninjected errors: caught {share(inj['caught'], inj['tried'])}; "
            f"the same changes uncorrupted flagged {share(inj['clean_flagged'], inj['tried'])}"
        )
        for kind, (k, n) in sorted(inj["kinds"].items()):
            print(f"    {kind:45s} {k}/{n}")
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
