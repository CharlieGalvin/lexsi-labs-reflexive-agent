"""Meta-healing audits the training data: the frozen checker on SFT v2's own training conversations.

    python scripts/data/audit_training_data.py                       # data/train/sft-v2.jsonl
    python scripts/data/audit_training_data.py --show 10             # print 10 flagged changes
    python scripts/data/audit_training_data.py --data data/train/sft-v4.jsonl

Every training conversation is the teacher's (GLM-5.3-Flash), verified by τ²-bench. τ²-bench
checks the end state, not how the agent got there, so a change the teacher made without the
customer's yes, or that doesn't match what they asked, still passes, and v2 learned from it. This
runs the checker the guard used ("no yes" and "mismatch", `stage2-checker-frozen-2`) over each
state-changing call in each training conversation, as the guard would have checked it live, and
reports how many conversations teach such a change (STAGE2.md, Stage 2b).

The conversations are read back from the training records' Cohere format into OpenAI-style
messages (`to_openai`); the policy the checker reads is the record's system prompt.
Writes ``results/audit/<data>-audit.jsonl`` (one line per flagged conversation) and
``results/audit/<data>-audit.json`` (the counts). Needs OPENAI_API_KEY; answers are cached
(results/cache/), so a re-run is free. Train-task records only: any other split is refused.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from reflexive.files import ROOT, read_jsonl
from reflexive.splits import split_of
from reflexive.teacher import parse_prompted

DATA = ROOT / "data" / "train" / "sft-v2.jsonl"
OUT = ROOT / "results" / "audit"
CACHE = ROOT / "results" / "cache"
DOMAINS = ("retail", "airline", "telecom")
_RESULT = re.compile(r"<results>\nDocument: (\d+)\ntool_name: (.*?)\nresult: (.*?)\n</results>", re.DOTALL)

Record = dict[str, Any]
Findings = Callable[[Record], list[dict]]


def to_openai(msgs: list[dict]) -> list[dict]:
    """A training record's Cohere-format messages as OpenAI-style ones: ``Action:`` replies become
    ``tool_calls``, ``<results>`` turns become ``tool`` messages (matched to the calls in order), and
    any customer text merged into a results turn follows as its own ``user`` message."""
    out: list[dict] = []
    pending: list[dict] = []
    n = 0
    for m in msgs:
        role, text = m["role"], m.get("content") or ""
        if role == "assistant" and text.startswith("Action: ```json"):
            calls = parse_prompted(text) or []
            for c in calls:
                c["id"] = f"c{n}"
                n += 1
            pending = list(calls)
            out.append({"role": "assistant", "content": "", "tool_calls": calls})
        elif role == "user" and text.startswith("<results>"):
            found = _RESULT.findall(text)
            for (_, name, result), call in zip(found, pending, strict=False):
                out.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
            pending = []
            rest = _RESULT.sub("", text).strip()
            if rest:
                out.append({"role": "user", "content": rest})
        else:
            out.append({"role": role, "content": text})
    return out


def checker_findings(rec: Record) -> list[dict]:
    """The frozen checker's live findings ("no yes", then "mismatch") on every change in ``rec``."""
    from reflexive.intent import intent_findings, no_yes
    from reflexive.mismatch import mismatch

    return intent_findings(rec, rules=[no_yes, mismatch])


def audit(examples: Iterable[Record], findings: Findings = checker_findings) -> tuple[dict, list[dict]]:
    """Counts per domain and the flagged conversations. Checker errors never flag (as in the guard)
    and are counted apart."""
    rows: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    flagged: list[dict] = []
    for ex in examples:
        d, t = ex["domain"], str(ex["task_id"])
        if split_of(d, t) != "train":
            raise SystemExit(f"⛔ {ex.get('id')}: a {split_of(d, t)} task; only train tasks are audited")
        fs = findings({"domain": d, "task_id": t, "agent_view": to_openai(ex["messages"])})
        live = [f for f in fs if f.get("live")]
        errors = [f for f in fs if not f.get("live")]
        for row in (rows[d], rows["all"]):
            row["conversations"] += 1
            row["flagged"] += bool(live)
            row["no_yes"] += any(f.get("rule") == "no_yes" for f in live)
            row["mismatch"] += any(f.get("rule") == "mismatch" for f in live)
            row["checker_errors"] += bool(errors)
        if live:
            flagged.append(
                {
                    "id": ex.get("id"),
                    "group": ex.get("group"),
                    "domain": d,
                    "task_id": t,
                    "findings": [{k: f.get(k) for k in ("turn", "tool", "rule", "reason")} for f in live],
                }
            )
    return {d: dict(rows[d]) for d in [*DOMAINS, "all"] if d in rows}, flagged


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--groups", nargs="*", help="only these example groups (default: all)")
    p.add_argument("--show", type=int, default=0, help="print this many flagged changes")
    a = p.parse_args()
    os.environ["REFLEXIVE_INTENT_JUDGE"] = "on"
    os.environ.setdefault("REFLEXIVE_JUDGE_CACHE", str(CACHE / "intent-judge.jsonl"))
    exs = [r for r in read_jsonl(a.data) if not a.groups or r.get("group") in a.groups]
    counts, flagged = audit(exs)
    for d, row in counts.items():
        n = row["conversations"]
        print(
            f"{d:<8} {row['flagged']}/{n} training conversations teach a flagged change"
            f" ({row['flagged'] / n:.0%}): no_yes {row['no_yes']}, mismatch {row['mismatch']}"
            + (f" | ⚠ checker errors in {row['checker_errors']}" if row["checker_errors"] else "")
        )
    for f in flagged[: a.show]:
        for x in f["findings"]:
            print(f"\n--- {f['id']} turn {x['turn']} {x['tool']} [{x['rule']}]\n    {str(x['reason'])[:400]}")
    OUT.mkdir(parents=True, exist_ok=True)
    stem = a.data.stem
    (OUT / f"{stem}-audit.jsonl").write_text("".join(json.dumps(f) + "\n" for f in flagged))
    (OUT / f"{stem}-audit.json").write_text(
        json.dumps({"data": str(a.data), "counts": counts}, indent=1) + "\n"
    )
    print(f"\nsaved results/audit/{stem}-audit.json and -audit.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
