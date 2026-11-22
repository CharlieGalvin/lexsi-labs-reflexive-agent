"""Meta-healing v2 (STAGE2.md, Sat 21 Nov 22:01): learn on the train pool, test once on fresh conversations.

    just meta-heal-v2 learn                  # up to 3 rounds on S2-*-trainpool.jsonl (~$25)
    just meta-heal-v2 score                  # the last round, once, on S2-*-fresh.jsonl (~$40)
    just meta-heal-v2 review                 # 5 flagged conversations per rule that meets the bar
    just meta-heal-v2 accept --names a,b --reviewer Charlie   # → configs/meta_rules.json

The learning set is SFT v2's 112 train-pool conversations. The held-out set is fresh conversations
on the same tasks (run tonight; see STAGE2.md). Every conversation goes through `assert_not_test`.
Needs OPENAI_API_KEY and litellm; every answer is cached in results/cache/, so a re-run is free.

Writes ``results/meta-v2/`` (with conversation text: local only) and
``results/meta-heal-v2.metrics.json`` (ids, rules and counts only: committed).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

from reflexive import meta_heal as mh
from reflexive import meta_heal_v2 as v2
from reflexive.hallucination import FailureType, agent_view_of, classify_record
from reflexive.metrics import is_broken
from reflexive.splits import assert_not_test, split_of
from reflexive.stats import wilson

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "meta-v2"
METRICS = ROOT / "results" / "meta-heal-v2.metrics.json"
CACHE = ROOT / "results" / "cache"
DOMAINS = ("retail", "airline", "telecom")
SETUP_ERROR = re.compile(
    r"Missing credentials|AuthenticationError|api_key|ModuleNotFoundError|No module named"
)


def load(episodes: Path, split: str) -> list[dict]:
    recs: dict[tuple, dict] = {}
    for d in DOMAINS:
        f = episodes / f"S2-{d}-{split}.jsonl"
        if not f.exists():
            print(f"⚠ {f.name} not found: left out")
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue
            if r and not is_broken(r):
                assert_not_test(d, str(r["task_id"]))
                if split_of(d, str(r["task_id"])) != "train":
                    sys.exit(f"⛔ {d} task {r['task_id']} isn't a train task")
                recs[(d, str(r["task_id"]), r.get("sample", 0))] = r
    if not recs:
        sys.exit(f"no S2-*-{split}.jsonl conversations in {episodes}")
    return list(recs.values())


def write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")


def read(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"{p.name} not found: run the step before this one")
    return json.loads(p.read_text())


def checker() -> mh.LLM:
    return mh.LLM(v2.CHECKER_MODEL, cache_path=CACHE / "meta-v2-checker.jsonl")


def stop_on_setup_errors(found: dict) -> None:
    for c in found.values():
        for fs in c.values():
            for f in fs:
                if f.get("error") and SETUP_ERROR.search(f["error"]):
                    sys.exit(
                        f"⛔ the checker can't run: {f['error']}\n"
                        "  Fix it (OPENAI_API_KEY, litellm) and run again."
                    )


def cmd_learn(a: argparse.Namespace) -> int:
    recs = load(a.episodes, "trainpool")
    print(f"learning set: {len(recs)} conversations")
    proposer = mh.LLM(v2.PROPOSER_MODEL, cache_path=CACHE / "meta-v2-proposer.jsonl")
    chk = checker()
    result = v2.learn(recs, proposer, chk, rounds=a.rounds, on_found=stop_on_setup_errors)
    print(f"\n{proposer.calls} proposer and {chk.calls} checker calls")
    write(
        OUT / "learn.json", {"proposer_model": v2.PROPOSER_MODEL, "checker_model": v2.CHECKER_MODEL, **result}
    )
    print("\nfinal rules (scored once on the held-out set next):")
    for c in result["final"]:
        print(
            f"  {c['name']} [{c['applies_to']}, {'live' if c['live'] else 'after the fact'}] "
            f"{c['tools']}\n    {c['what']}"
        )
    return 0


def frozen_findings(recs: list[dict]) -> tuple[dict, list[str]]:
    from reflexive.intent import intent_findings, no_yes
    from reflexive.mismatch import mismatch

    out, errors = {}, []
    for r in recs:
        ny = intent_findings(r, rules=[no_yes])
        mm = intent_findings(r, rules=[mismatch])
        errs = [f["reason"] for f in ny + mm if not f["live"]]
        if setup := next((e for e in errs if SETUP_ERROR.search(e)), None):
            sys.exit(f"⛔ the checker can't run: {setup}")
        errors += errs
        out[mh.example_id(r)] = {
            "tal": classify_record(r).failure != FailureType.NONE,
            "no_yes": any(f["live"] for f in ny),
            "mismatch": any(f["live"] for f in mm),
        }
    return out, errors


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n) if n else (0.0, 0.0)
    return f"{k}/{n} {(k / n if n else 0):.0%} [{lo:.0%}–{hi:.0%}]"


def cmd_score(a: argparse.Namespace) -> int:
    learned = read(OUT / "learn.json")
    cands = [mh.Candidate.from_dict(c) for c in learned["final"]]
    recs = load(a.episodes, "fresh")
    print(f"held-out set: {len(recs)} fresh conversations; {len(cands)} rules, scored once")
    frozen, ferrs = frozen_findings(recs)
    fails = [r for r in recs if not mh.passed(r)]
    passes = [r for r in recs if mh.passed(r)]
    print("\nthe frozen rules on the held-out set (their first out-of-sample rates):")
    rates = {}
    for key in ("no_yes", "mismatch"):
        fk = sum(frozen[mh.example_id(r)][key] for r in fails)
        pk = sum(frozen[mh.example_id(r)][key] for r in passes)
        rates[key] = {"failures": [fk, len(fails)], "passes": [pk, len(passes)]}
        print(f"  {key:9s} failures {pct(fk, len(fails))}   passes {pct(pk, len(passes))}")
    if ferrs:
        print(f"  ⚠ {len(ferrs)} checker errors on the frozen rules, counted as not flagged")
    chk = checker()
    scores = v2.held_out(cands, recs, frozen, chk, on_found=stop_on_setup_errors)
    n_added = scores[0].added[1] if scores else 0
    print(
        f"\nbar: ≥{v2.MIN_ADDED} and ≥{v2.MIN_ADDED_SHARE:.0%} of the {n_added} failures "
        "no frozen rule catches; "
        f"≤{v2.MAX_PASS_SHARE:.0%} of passes, at worst\n"
    )
    for s in scores:
        print(
            f"  {'✓' if s.meets_bar else '✗'} {s.name:45s} added {pct(*s.added)}  passes {pct(*s.passes)}"
            + (f" (at worst {s.passes[0] + s.passes_errored})" if s.passes_errored else "")
            + f"  rediscovers mismatch {s.rediscovers[0]}/{s.rediscovers[1]}"
            + (f"  ⚠ {s.errors} errors" if s.errors else "")
        )
    short = v2.shortlist(scores)
    print(
        f"\nmeets the bar (best {v2.MAX_ACCEPTED}): {', '.join(short) or 'none'}"
        f"  ({chk.calls} new checker calls)"
    )
    ids = {mh.example_id(r): mh.passed(r) for r in recs}
    table = v2.layers(ids, frozen, [x.to_dict() for x in scores], short)
    print_layers(table, "if every rule that meets the bar is accepted")
    write(
        OUT / "scores.json",
        {
            "shortlist": short,
            "scores": [s.to_dict() for s in scores],
            "frozen_rates": rates,
            "frozen": frozen,
            "ids": ids,
        },
    )
    write_metrics(
        learned, scores_d=[s.to_dict() for s in scores], short=short, rates=rates, accepted=None, layers=table
    )
    return 0


def print_layers(table: list[dict], note: str) -> None:
    print(f"\nwhat the loop sees without an answer key, on the held-out set ({note}):")
    for row in table:
        print(f"  {row['layer']:60s} failures {pct(*row['failures'])}   passes {pct(*row['passes'])}")


def write_metrics(
    learned: dict,
    scores_d: list[dict],
    short: list[str],
    rates: dict,
    accepted: dict | None,
    layers: list[dict] | None = None,
) -> None:
    held = [
        {
            k: s[k]
            for k in (
                "name",
                "added",
                "added_share",
                "failures",
                "passes",
                "passes_errored",
                "pass_share_at_worst",
                "rediscovers",
                "errors",
                "meets_bar",
            )
        }
        | {"flagged_ids": [{"id": x["id"], "kind": x["kind"]} for x in s["flagged"]]}
        for s in scores_d
    ]
    write(
        METRICS,
        {
            "ticket": "SEL-91 (v2)",
            "proposer_model": v2.PROPOSER_MODEL,
            "checker_model": v2.CHECKER_MODEL,
            "bar": {
                "min_added": v2.MIN_ADDED,
                "min_added_share": v2.MIN_ADDED_SHARE,
                "max_pass_share": v2.MAX_PASS_SHARE,
                "review": "≥4 of 5 real errors",
            },
            "learning": {
                k: learned[k] for k in ("unexplained_failures", "passes_shown", "passes_scored", "rounds")
            },
            "final": learned["final"],
            "held_out": held,
            "frozen_rules_held_out": rates,
            "layers": layers,
            "meets_bar": short,
            "accepted": accepted,
        },
    )


def cmd_review(a: argparse.Namespace) -> int:
    sc = read(OUT / "scores.json")
    recs = {mh.example_id(r): r for r in load(a.episodes, "fresh")}
    names = a.only.split(",") if a.only else sc["shortlist"]
    if not names:
        print("no rule meets the bar: nothing to review. Record the result with `accept` and no names.")
        return 0
    order = {"added": 0, "failure": 1, "pass": 2}
    for name in names:
        s = next(x for x in sc["scores"] if x["name"] == name)
        flagged = sorted(s["flagged"], key=lambda x: order[x["kind"]])[: v2.REVIEW_EXAMPLES]
        lines = [
            f"# Review: {name}\n",
            "Is each flagged moment a real error of the kind the rule names? yes / no\n",
            "Also: is the rule general, or does it only work by knowing these tasks?\n",
        ]
        for x in flagged:
            view = agent_view_of(recs[x["id"]])
            lines.append(f"\n## {x['id']} ({x['kind']})\n")
            for f in x["findings"]:
                lines.append(
                    f'- turn {f["turn"]} {f["tool"] or ""}: evidence "{f["evidence"]}". {f["reason"]}'
                )
                lo = max(1, f["turn"] - 8)
                lines.append("\n```\n" + mh.transcript(view[lo : f["turn"] + 1], 800, 3000) + "\n```\n")
            lines.append("Real error? ____\n")
        p = OUT / f"review-{name}.md"
        p.write_text("\n".join(lines))
        print(f"wrote {p.relative_to(ROOT)} ({len(flagged)} conversations)")
    return 0


def cmd_accept(a: argparse.Namespace) -> int:
    learned, sc = read(OUT / "learn.json"), read(OUT / "scores.json")
    names = [n for n in (a.names or "").split(",") if n]
    if bad := [n for n in names if n not in sc["shortlist"]]:
        sys.exit(f"not on the shortlist: {bad}")
    if names and not a.reviewer:
        sys.exit("--reviewer: who read the 5 conversations for each rule")
    rules = [c for c in learned["final"] if c["name"] in names]
    accepted = {"names": names, "reviewer": a.reviewer, "date": dt.date.today().isoformat()}
    write(mh.RULES_PATH, {"ticket": "SEL-91", "v1_accepted": [], "version": 2, **accepted, "accepted": rules})
    table = v2.layers(sc["ids"], sc["frozen"], sc["scores"], names)
    print_layers(table, "with the accepted rules")
    write_metrics(learned, sc["scores"], sc["shortlist"], sc["frozen_rates"], accepted, layers=table)
    print(f"\naccepted: {', '.join(names) or 'none (that is the result)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["learn", "score", "review", "accept"])
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--rounds", type=int, default=v2.ROUNDS)
    p.add_argument("--only", help="review only these rules (comma-separated)")
    p.add_argument("--names", help="accept: the reviewed rules (comma-separated; none is allowed)")
    p.add_argument("--reviewer")
    a = p.parse_args(argv)
    if a.rounds > v2.ROUNDS:
        sys.exit(f"STAGE2.md allows at most {v2.ROUNDS} rounds")
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    os.environ.setdefault("REFLEXIVE_META_CACHE", str(CACHE / "meta-rules.jsonl"))
    return {"learn": cmd_learn, "score": cmd_score, "review": cmd_review, "accept": cmd_accept}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
