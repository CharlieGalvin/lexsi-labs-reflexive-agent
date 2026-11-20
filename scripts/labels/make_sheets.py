"""Build the blind labelling sheets for SEL-85 (classifier audit) and SEL-87 (the 37), on the pod.

    python scripts/labels/make_sheets.py                    # → results/labels/ (git-ignored)
    python scripts/labels/make_sheets.py --labellers katie charlie --seed 0

Reads the saved Stage 1 conversations (results/episodes/arms/, base model P and SFT v2 S2, dev and
validation only, never test), runs the Tool/Arg/Loop classifier (`reflexive.hallucination`), and
writes three sheets, each in one copy per labeller so you label separately:

1. **audit-precision**: ~20 flagged findings per type (Tool, Arg, Loop), one per conversation.
   Each says what the classifier claims; you answer "is this really a hallucination?" (yes/no).
2. **audit-recall**: ~40 random conversations, flagged or not. The classifier's verdict is hidden;
   you answer "does this conversation contain a hallucination?" (yes/no) and which type(s).
3. **decisions**: SFT v2's failed dev and validation conversations with no classifier finding
   (the 37). You choose a category from the labelling guide (SEL-81).

Every item is shuffled, gets an anonymous ID (P01…, R01…, D01…) and a readable transcript in
``<sheet>/items/<ID>.md``. Which model and task an ID is, and the classifier's verdict, are in
``<sheet>-key.csv``: don't open it until both of you have finished. Scoring afterwards:
`just label-metrics` (see that script's docstring). Re-running with the same seed gives the same
sheets; it won't overwrite a labeller's sheet that's already been filled in.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from reflexive.hallucination import FailureType, agent_view_of, classify_record
from reflexive.metrics import is_broken

ROOT = Path(__file__).resolve().parents[2]
EPISODES = ROOT / "results" / "episodes" / "arms"
OUT = ROOT / "results" / "labels"
DOMAINS = ("retail", "airline", "telecom")
MODELS = {"P": "base", "S2": "sft2"}
SPLITS = ("dev", "val")
TYPES = (FailureType.TOOL, FailureType.ARG, FailureType.LOOP)
TRUNC = 700  # characters of a tool result or long message shown in a transcript


def load(episodes: Path) -> list[dict]:
    """Scored conversations from the base model and v2 on dev and validation, with where they came from."""
    recs, missing = [], []
    for arm in MODELS:
        for d in DOMAINS:
            for s in SPLITS:
                f = episodes / f"{arm}-{d}-{s}.jsonl"
                if not f.exists():
                    missing.append(f.name)
                    continue
                for line in f.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        if not is_broken(r):
                            recs.append({**r, "_arm": arm, "_split": s, "_file": f.name})
    if missing:
        print(f"⚠ not found (left out): {', '.join(missing)}")
    return recs


def short(text: object, n: int = TRUNC) -> str:
    s = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + f" … [{len(s) - n} more characters]"


def transcript(rec: dict, mark: int | None = None, note: str = "") -> str:
    """The conversation as the agent saw it, turn by turn; ``mark`` highlights one message index."""
    lines = [f"Domain: {rec['domain']}\n"]
    if note:
        lines.append(note + "\n")
    for i, m in enumerate(agent_view_of(rec)):
        role = m.get("role")
        flag = "  ◀◀◀ FLAGGED TURN" if i == mark else ""
        if role == "system":
            lines.append(
                f"[{i}] SYSTEM (the policy and instructions;"
                f" {len(m.get('content') or '')} characters, not shown)"
            )
            continue
        who = {"user": "CUSTOMER", "assistant": "AGENT", "tool": "TOOL RESULT"}.get(role, str(role).upper())
        body = short(m.get("content")) if m.get("content") else ""
        lines.append(f"[{i}] {who}{flag}" + (f": {body}" if body else ""))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            lines.append(f"      → calls {fn.get('name')}({short(fn.get('arguments'), 400)})")
    if rec.get("termination"):
        lines.append(f"\n(Conversation ended: {rec['termination']})")
    return "\n".join(lines) + "\n"


def write_sheet(out: Path, name: str, columns: list[str], rows: list[dict], labellers: list[str]) -> None:
    for who in labellers:
        f = out / f"{name}-{who}.csv"
        if f.exists() and any(
            any(v for k, v in r.items() if k in ("label", "category", "notes"))
            for r in csv.DictReader(f.open())
        ):
            print(f"  kept {f.name}: already has labels in it")
            continue
        with f.open("w", newline="") as h:
            w = csv.DictWriter(h, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in columns})


def write_key(out: Path, name: str, rows: list[dict]) -> None:
    cols = ["id", "model", "split", "domain", "task_id", "sample", "classifier", "classifier_types", "passed"]
    with (out / f"{name}-key.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def key_row(item_id: str, rec: dict, diag) -> dict:  # noqa: ANN001
    return {
        "id": item_id,
        "model": MODELS[rec["_arm"]],
        "split": rec["_split"],
        "domain": rec["domain"],
        "task_id": rec["task_id"],
        "sample": rec.get("sample", ""),
        "classifier": "yes" if diag.findings else "no",
        "classifier_types": " ".join(sorted(t.value for t in diag.types)),
        "passed": "yes" if rec["outcome"]["passed"] else "no",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labellers", nargs="+", default=["katie", "charlie"])
    p.add_argument("--per-type", type=int, default=20, help="flagged findings per type (precision)")
    p.add_argument("--recall", type=int, default=40, help="random conversations (recall)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=Path, default=EPISODES)
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument(
        "--expect-decisions", type=int, default=37, help="warn if the decision pool isn't this size"
    )
    a = p.parse_args(argv)
    rng = random.Random(a.seed)

    recs = load(a.episodes)
    if not recs:
        raise SystemExit(f"no conversations found in {a.episodes}")
    diags = [classify_record(r) for r in recs]
    print(f"{len(recs)} scored conversations (base and v2, dev and validation)")

    # 1. Precision: per type, conversations with a finding of that type; one finding each
    prec_items = []
    for t in TYPES:
        pool = [(r, d) for r, d in zip(recs, diags, strict=True) if t in d.types]
        pick = rng.sample(pool, min(a.per_type, len(pool)))
        print(f"  precision, {t.value}: {len(pick)} of {len(pool)} conversations with a {t.value} finding")
        for r, d in pick:
            f = rng.choice([x for x in d.findings if x.type == t])
            prec_items.append((r, d, f))
    rng.shuffle(prec_items)

    # 2. Recall: random conversations from the whole pool
    rec_items = rng.sample(list(zip(recs, diags, strict=True)), min(a.recall, len(recs)))

    # 3. Decisions: v2's failed conversations with no finding
    dec_items = [
        (r, d)
        for r, d in zip(recs, diags, strict=True)
        if r["_arm"] == "S2" and not r["outcome"]["passed"] and not d.findings
    ]
    rng.shuffle(dec_items)
    if len(dec_items) != a.expect_decisions:
        print(
            f"⚠ decision pool is {len(dec_items)}, not {a.expect_decisions}:"
            " check the episode files before labelling"
        )
    else:
        print(f"  decisions: {len(dec_items)} (the 37) ✓")

    a.out.mkdir(parents=True, exist_ok=True)
    sheets = {
        "audit-precision": (
            ["id", "claimed_type", "claimed_kind", "flagged_turn", "classifier_flagged", "label", "notes"],
            [(f"P{i:02d}", r, d, f) for i, (r, d, f) in enumerate(prec_items, 1)],
        ),
        "audit-recall": (
            ["id", "label", "types_seen", "notes"],
            [(f"R{i:02d}", r, d, None) for i, (r, d) in enumerate(rec_items, 1)],
        ),
        "decisions": (
            ["id", "category", "notes"],
            [(f"D{i:02d}", r, d, None) for i, (r, d) in enumerate(dec_items, 1)],
        ),
    }
    for name, (cols, items) in sheets.items():
        folder = a.out / name / "items"
        folder.mkdir(parents=True, exist_ok=True)
        rows, keys = [], []
        for item_id, r, d, f in items:
            if f is not None:
                claim = f"The classifier says this turn is **{f.type.value}** ({f.kind}): {f.detail or '—'}"
                text = transcript(r, mark=f.turn, note=claim)
                rows.append(
                    {
                        "id": item_id,
                        "claimed_type": f.type.value,
                        "claimed_kind": f.kind,
                        "flagged_turn": f.turn,
                        "classifier_flagged": "yes",
                    }
                )
            elif name == "decisions":
                checks = json.dumps(r["outcome"].get("checks"), ensure_ascii=False)
                text = transcript(r, note=f"τ²-bench: FAILED. Checks: {short(checks, 1500)}")
                rows.append({"id": item_id})
            else:
                text = transcript(r)
                rows.append({"id": item_id})
            (folder / f"{item_id}.md").write_text(f"# {item_id}\n\n" + text)
            keys.append(key_row(item_id, r, d))
        write_sheet(a.out, name, cols, rows, a.labellers)
        write_key(a.out, name, keys)
        print(f"wrote {name}: {len(items)} items, sheets for {', '.join(a.labellers)}, key in {name}-key.csv")
    print(
        f"\nAll in {a.out}. Label in the CSVs (label: yes/no; category: from the guide);"
        " keep the keys closed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
