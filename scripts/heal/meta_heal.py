"""SEL-91, meta-healing: the checker proposes its own rules, scores them on held-out tasks, a person accepts.

    just meta-heal split                      # the 2/3 : 1/3 split and the unexplained failures; no API call
    just meta-heal propose                    # GPT-4.1 writes ≤10 candidates from the propose set (~$1)
    just meta-heal score                      # each candidate on the held-out third (gpt-4.1-mini, ~$2)
    just meta-heal review                     # 5 flagged conversations per candidate that meets the bar
    just meta-heal accept --names a,b --reviewer Charlie   # after reading them: configs/meta_rules.json
    just meta-heal propose --round 2          # a further round: the earlier rules are shown as tried,
    just meta-heal score --round 2            #   and the 10 scored candidates are shared by all rounds

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
    print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")


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


def paths(n: int) -> tuple[Path, Path]:
    """Round 1 keeps the plain names (it ran before rounds existed); round n writes ``-r<n>``."""
    tag = "" if n == 1 else f"-r{n}"
    return OUT / f"candidates{tag}.json", OUT / f"scores{tag}.json"


def rounds() -> list[int]:
    return [n for n in range(1, 10) if paths(n)[0].exists()]


def cmd_propose(recs: list[dict], a: argparse.Namespace) -> int:
    n = a.round
    earlier = [r for r in rounds() if r < n]
    if n > 1 and earlier != list(range(1, n)):
        sys.exit(f"round {n} needs rounds 1–{n - 1} proposed first")
    tried = [mh.Candidate.from_dict(c) for r in earlier for c in read(paths(r)[0])["candidates"]]
    limit = mh.MAX_CANDIDATES - len(tried)
    if limit <= 0:
        sys.exit(
            f"{len(tried)} candidates already proposed: STAGE2.md allows {mh.MAX_CANDIDATES} scored in all"
        )
    prop, held = mh.split(recs, seed=a.seed)
    llm = mh.LLM(mh.PROPOSER_MODEL, cache_path=CACHE / "meta-proposer.jsonl")
    cands, meta = mh.propose(prop, llm, n_passes=a.passes, seed=a.seed, round=n, tried=tried, limit=limit)
    n_fail, n_pass = len(meta["unexplained_failures"]), len(meta["passes_shown"])
    print(f"round {n}: {n_fail} unexplained failures and {n_pass} passes shown")
    print(f"{meta['proposed']} proposed, {len(cands)} kept (at most {limit} this round):\n")
    for c in cands:
        mark = "live" if c.live else "after the fact"
        print(f"  {c.name} [{c.applies_to}, {mark}]\n    {c.what}\n    Q: {c.question}")
    write(
        paths(n)[0],
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
    cand = read(paths(a.round)[0])
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
        f"round {a.round}, held out: {len(held)} conversations; bar: ≥{mh.MIN_FLAGGED} and "
        f"≥{mh.MIN_SHARE:.0%} of unexplained failures, ≤{mh.MAX_PASSES} passes\n"
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
    new = {s.name: s.to_dict() for s in scores}
    if a.only and paths(a.round)[1].exists():  # keep the scores of the candidates not re-run
        old = {s["name"]: s for s in read(paths(a.round)[1])["scores"]}
        order = [c["name"] for c in cand["candidates"]]
        merged = {**old, **new}
        rows = [merged[n] for n in order if n in merged]
    else:
        rows = list(new.values())
    write(paths(a.round)[1], {"rule_model": mh.RULE_MODEL, "scores": rows})
    short = shortlist()
    print(f"meets the bar, all rounds (best {mh.MAX_ACCEPTED}): {', '.join(short) or 'none'}")
    write_metrics(accepted=recorded_acceptance())
    return 0


def recorded_acceptance() -> dict | None:
    """The accept already in configs/meta_rules.json (names, reviewer, date), so a re-score keeps it."""
    p = mh.rules_path()
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return {k: d[k] for k in ("names", "reviewer", "date")} if "names" in d else None


def all_scores() -> list[dict]:
    return [s for n in rounds() if paths(n)[1].exists() for s in read(paths(n)[1])["scores"]]


def shortlist() -> list[str]:
    """The best `MAX_ACCEPTED` that meet the bar, across every round."""
    return mh.shortlist([mh.Score(**{k: s[k] for k in mh.Score.__dataclass_fields__}) for s in all_scores()])


def write_metrics(accepted: dict | None) -> None:
    scores = {s["name"]: s for s in all_scores()}
    out_rounds = []
    for n in rounds():
        cand = read(paths(n)[0])
        rows = []
        for c in cand["candidates"]:
            row = {k: c[k] for k in ("name", "what", "applies_to", "question", "live")}
            if s := scores.get(c["name"]):
                row |= {
                    k: s[k]
                    for k in ("unexplained_failures", "other_failures", "passes", "share", "meets_bar")
                }
                row["flagged_ids"] = [
                    {"id": x["id"], "kind": x["kind"], "turns": [f["turn"] for f in x["findings"]]}
                    for x in s["flagged"]
                ]
            rows.append(row)
        out_rounds.append(
            {
                "round": n,
                "proposer_model": cand["proposer_model"],
                "proposed": cand["proposed"],
                "unexplained_failures_shown": cand["unexplained_failures"],
                "candidates": rows,
            }
        )
    first = read(paths(1)[0])
    write(
        METRICS,
        {
            "ticket": "SEL-91",
            "seed": first["seed"],
            "rule_model": mh.RULE_MODEL,
            "bar": {"min_flagged": mh.MIN_FLAGGED, "min_share": mh.MIN_SHARE, "max_passes": mh.MAX_PASSES},
            "max_candidates_scored": mh.MAX_CANDIDATES,
            "propose_conversations": len(first["propose_ids"]),
            "holdout_ids": first["holdout_ids"],
            "rounds": out_rounds,
            "meets_bar": shortlist(),
            "accepted": accepted,
        },
    )


def cmd_review(recs: list[dict], a: argparse.Namespace) -> int:
    scores = {s["name"]: s for s in all_scores()}
    by_id = {mh.example_id(r): r for r in recs}
    names = a.only.split(",") if a.only else shortlist()
    if not names:
        print("no candidate meets the bar: nothing to review. Record the result with `accept` and no names.")
        return 0
    for name in names:
        s = scores[name]
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
    short = shortlist()
    names = [n for n in (a.names or "").split(",") if n]
    if bad := [n for n in names if n not in short]:
        sys.exit(f"not on the shortlist (didn't meet the bar, or past the best {mh.MAX_ACCEPTED}): {bad}")
    if names and not a.reviewer:
        sys.exit("--reviewer: who read the 5 conversations for each rule")
    rules = [c for n in rounds() for c in read(paths(n)[0])["candidates"] if c["name"] in names]
    accepted = {"names": names, "reviewer": a.reviewer, "date": dt.date.today().isoformat()}
    write(mh.rules_path(), {"ticket": "SEL-91", "rule_model": mh.RULE_MODEL, **accepted, "accepted": rules})
    write_metrics(accepted)
    print(f"\naccepted: {', '.join(names) or 'none (that is the result)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["split", "propose", "score", "review", "accept"])
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--arm", default="S2")
    p.add_argument("--split", dest="split_name", default="trainpool")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--round", type=int, default=1, help="propose / score: which round (STAGE2: ≤10 scored in all)"
    )
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
