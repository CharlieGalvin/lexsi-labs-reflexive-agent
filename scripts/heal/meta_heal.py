"""SEL-91, meta-healing: the checker proposes its own rules, scores them on held-out tasks, a person accepts.

    just meta-heal split                      # the 2/3 : 1/3 split and the unexplained failures; no API call
    just meta-heal propose                    # GPT-4.1 writes ≤10 candidates from the propose set (~$1)
    just meta-heal score                      # each candidate on the held-out third (gpt-4.1-mini, ~$2)
    just meta-heal review                     # 5 flagged conversations per candidate that meets the bar
    just meta-heal accept --names a,b --reviewer Charlie   # after reading them: configs/meta_rules.json

Reads SFT v2's train-task conversations (``results/episodes/arms/S2-*-trainpool.jsonl``); every
one goes through `assert_not_test`. Needs OPENAI_API_KEY (``set -a; source .env; set +a``) and
litellm. Every model answer is cached (``results/cache/meta-*.jsonl``), so re-running a step is free
and gives the same result.

Writes, in ``results/``:

* ``meta/candidates.json``, ``meta/scores.json``, ``meta/review-<rule>.md``: local, with conversation
  text (never committed);
* ``meta-heal.metrics.json``: committed. The split, every candidate and its held-out counts,
  whether it met the bar, which were accepted; ids and turns only, no conversation text.

The bar (STAGE2.md): on the held-out third, a candidate flags at least 2, and at least 20%, of the
unexplained failures (no Tool / Arg / Loop or "no yes" finding), and at most 1 pass; at most 10
scored, at most 3 accepted, each only after a person reads 5 conversations it flags. If none passes,
that is the result: ``accept`` with no names writes an empty rule list and the metrics say so.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

from reflexive import meta_heal as mh
from reflexive.hallucination import agent_view_of
from reflexive.metrics import is_broken
from reflexive.splits import assert_not_test

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "meta"
METRICS = ROOT / "results" / "meta-heal.metrics.json"
CACHE = ROOT / "results" / "cache"
DOMAINS = ("retail", "airline", "telecom")


def load(episodes: Path, arm: str, split: str) -> list[dict]:
    recs: dict[tuple, dict] = {}
    for d in DOMAINS:
        f = episodes / f"{arm}-{d}-{split}.jsonl"
        if not f.exists():
            print(f"⚠ {f.name} not found: left out")
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line) if line.strip() else None
            except json.JSONDecodeError:
                continue  # a half-written last line
            if r and not is_broken(r):
                assert_not_test(d, str(r["task_id"]))
                recs[(d, str(r["task_id"]), r.get("sample", 0))] = r  # the latest copy of each
    return list(recs.values())


def read(p: Path) -> dict:
    if not p.exists():
        sys.exit(f"{p.relative_to(ROOT)} not found: run the step before this one")
    return json.loads(p.read_text())


def write(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {p.relative_to(ROOT)}")


def counts(recs: list[dict]) -> dict:
    out = {}
    for d in DOMAINS:
        rs = [r for r in recs if r["domain"] == d]
        fails = [r for r in rs if not mh.passed(r)]
        out[d] = {
            "conversations": len(rs),
            "passes": len(rs) - len(fails),
            "failures": len(fails),
            "unexplained_failures": sum(not mh.explained(r) for r in fails),
        }
    return out


def cmd_split(recs: list[dict], a: argparse.Namespace) -> int:
    prop, held = mh.split(recs, seed=a.seed)
    print(f"{len(recs)} conversations: propose {len(prop)}, held out {len(held)} (seed {a.seed})\n")
    for name, rs in (("propose", prop), ("held out", held)):
        for d, c in counts(rs).items():
            print(f"  {name:8s} {d:8s} {c}")
    return 0


def cmd_propose(recs: list[dict], a: argparse.Namespace) -> int:
    prop, held = mh.split(recs, seed=a.seed)
    llm = mh.LLM(mh.PROPOSER_MODEL, cache_path=CACHE / "meta-proposer.jsonl")
    cands, meta = mh.propose(prop, llm, n_passes=a.passes, seed=a.seed)
    n_fail, n_pass = len(meta["unexplained_failures"]), len(meta["passes_shown"])
    print(f"{n_fail} unexplained failures and {n_pass} passes shown")
    print(f"{meta['proposed']} proposed, {len(cands)} kept (at most {mh.MAX_CANDIDATES}):\n")
    for c in cands:
        mark = "live" if c.live else "after the fact"
        print(f"  {c.name} [{c.applies_to}, {mark}]\n    {c.what}\n    Q: {c.question}")
    write(
        OUT / "candidates.json",
        {
            "seed": a.seed,
            "proposer_model": mh.PROPOSER_MODEL,
            "propose_ids": [mh.example_id(r) for r in prop],
            "holdout_ids": [mh.example_id(r) for r in held],
            **meta,
            "candidates": [c.to_dict() for c in cands],
        },
    )
    return 0


def cmd_score(recs: list[dict], a: argparse.Namespace) -> int:
    cand = read(OUT / "candidates.json")
    _, held = mh.split(recs, seed=cand["seed"])
    if [mh.example_id(r) for r in held] != cand["holdout_ids"]:
        sys.exit("the held-out conversations differ from those at propose time: same episodes, please")
    cands = [mh.Candidate.from_dict(c) for c in cand["candidates"]]
    if a.only:
        cands = [c for c in cands if c.name in a.only.split(",")]
    llm = mh.LLM(mh.RULE_MODEL, cache_path=CACHE / "meta-rules.jsonl")
    scores = mh.score(cands, held, llm)
    errors = sum(s.errors for s in scores)
    print(
        f"held out: {len(held)} conversations; bar: ≥{mh.MIN_FLAGGED} and ≥{mh.MIN_SHARE:.0%} of unexplained "
        f"failures, ≤{mh.MAX_PASSES} passes\n"
    )
    for s in scores:
        u, o, p = s.unexplained_failures, s.other_failures, s.passes
        print(
            f"  {'✓' if s.meets_bar else '✗'} {s.name:40s} unexplained {u[0]}/{u[1]} ({s.share:.0%})"
            f"  explained {o[0]}/{o[1]}  passes {p[0]}/{p[1]}"
            + (f"  ⚠ {s.errors} errors" if s.errors else "")
        )
    print(f"\n{llm.calls} new model calls")
    if errors:
        first = next(s.first_error for s in scores if s.first_error)
        print(
            f"⚠ {errors} checker errors: those calls weren't flagged, so the counts are too low. "
            f"Fix it (usually OPENAI_API_KEY) and run again; nothing is written. First: {first}"
        )
        return 1
    short = mh.shortlist(scores)
    print(f"meets the bar (best {mh.MAX_ACCEPTED}): {', '.join(short) or 'none'}")
    write(
        OUT / "scores.json",
        {"rule_model": mh.RULE_MODEL, "shortlist": short, "scores": [s.to_dict() for s in scores]},
    )
    write_metrics(cand, scores, short, accepted=None)
    return 0


def write_metrics(cand: dict, scores: list, short: list[str], accepted: dict | None) -> None:
    rows = []
    for c in cand["candidates"]:
        s = next((x for x in scores if (x.name if hasattr(x, "name") else x["name"]) == c["name"]), None)
        s = s.to_dict() if hasattr(s, "to_dict") else s
        row = {k: c[k] for k in ("name", "what", "applies_to", "question", "live")}
        if s:
            row |= {
                k: s[k] for k in ("unexplained_failures", "other_failures", "passes", "share", "meets_bar")
            }
            row["flagged_ids"] = [
                {"id": x["id"], "kind": x["kind"], "turns": [f["turn"] for f in x["findings"]]}
                for x in s["flagged"]
            ]
        rows.append(row)
    write(
        METRICS,
        {
            "ticket": "SEL-91",
            "seed": cand["seed"],
            "proposer_model": cand["proposer_model"],
            "rule_model": mh.RULE_MODEL,
            "bar": {"min_flagged": mh.MIN_FLAGGED, "min_share": mh.MIN_SHARE, "max_passes": mh.MAX_PASSES},
            "propose_conversations": len(cand["propose_ids"]),
            "holdout_ids": cand["holdout_ids"],
            "unexplained_failures_shown": cand["unexplained_failures"],
            "candidates": rows,
            "meets_bar": short,
            "accepted": accepted,
        },
    )


def cmd_review(recs: list[dict], a: argparse.Namespace) -> int:
    sc = read(OUT / "scores.json")
    by_id = {mh.example_id(r): r for r in recs}
    names = a.only.split(",") if a.only else sc["shortlist"]
    if not names:
        print("no candidate meets the bar: nothing to review. Record the result with `accept` and no names.")
        return 0
    for name in names:
        s = next(x for x in sc["scores"] if x["name"] == name)
        flagged = sorted(s["flagged"], key=lambda x: x["kind"] != "unexplained_failures")[
            : mh.REVIEW_EXAMPLES
        ]
        lines = [
            f"# Review: {name}\n",
            "For each conversation: is the flagged moment a real error? yes / no\n",
        ]
        for x in flagged:
            r = by_id[x["id"]]
            view = agent_view_of(r)
            lines.append(f"\n## {x['id']} ({x['kind']}, {'passed' if mh.passed(r) else 'failed'})\n")
            for f in x["findings"]:
                lines.append(
                    f'- turn {f["turn"]} {f["tool"] or ""}: evidence "{f["evidence"]}". {f["reason"]}'
                )
                lo = max(1, f["turn"] - 8)
                lines.append("\n```\n" + mh.transcript(view[lo : f["turn"] + 1], 800) + "\n```\n")
            lines.append("Real error? ____\n")
        p = OUT / f"review-{name}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(lines))
        print(f"wrote {p.relative_to(ROOT)} ({len(flagged)} conversations)")
    return 0


def cmd_accept(recs: list[dict], a: argparse.Namespace) -> int:
    cand = read(OUT / "candidates.json")
    sc = read(OUT / "scores.json")
    names = [n for n in (a.names or "").split(",") if n]
    if bad := [n for n in names if n not in sc["shortlist"]]:
        sys.exit(f"not on the shortlist (didn't meet the bar, or past the best {mh.MAX_ACCEPTED}): {bad}")
    if names and not a.reviewer:
        sys.exit("--reviewer: who read the 5 conversations for each rule")
    rules = [c for c in cand["candidates"] if c["name"] in names]
    accepted = {"names": names, "reviewer": a.reviewer, "date": dt.date.today().isoformat()}
    write(mh.RULES_PATH, {"ticket": "SEL-91", "rule_model": mh.RULE_MODEL, **accepted, "accepted": rules})
    write_metrics(cand, sc["scores"], sc["shortlist"], accepted)
    print(f"\naccepted: {', '.join(names) or 'none (that is the result)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "propose", "score", "review", "accept"])
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--arm", default="S2")
    p.add_argument("--split", dest="split_name", default="trainpool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--passes", type=int, default=12, help="passes shown to the proposer")
    p.add_argument("--only", help="score or review only these candidates (comma-separated)")
    p.add_argument("--names", help="accept: the reviewed rules to accept (comma-separated; none is allowed)")
    p.add_argument("--reviewer", help="accept: who read the examples")
    a = p.parse_args(argv)
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"  # "explained" runs the "no yes" rule with its judge (cached)
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    recs = load(a.episodes, a.arm, a.split_name)
    if not recs:
        sys.exit(f"no conversations found in {a.episodes}")
    return {
        "split": cmd_split,
        "propose": cmd_propose,
        "score": cmd_score,
        "review": cmd_review,
        "accept": cmd_accept,
    }[a.step](recs, a)


if __name__ == "__main__":
    sys.exit(main())
