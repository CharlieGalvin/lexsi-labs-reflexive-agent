"""Spot-check the hallucination classifier (SEL-12): sample findings to label by hand.

    python scripts/spot_check_hallucination.py results/episodes/arms/*-telecom-train.jsonl
    python scripts/spot_check_hallucination.py FILES --labels docs/experiments/hallucination_labels.json

Samples ``--per-type`` findings (default 10) of each type, TOOL, ARG and LOOP, from **failed**
conversations, one finding per conversation, spread across the kinds within each type, with a
fixed seed. It prints each with the context needed to judge it: what the customer and the tools
had said before, the call, and its result. With ``--labels`` (a JSON object ``{"<sample id>":
true | false}``) it reports precision per type instead.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflexive.hallucination import FailureType, agent_view_of, classify_record  # noqa: E402

SEED = "sel-12-spot-check-v1"


def sample(paths: list[Path], per_type: int) -> list[dict]:
    rng = random.Random(SEED)
    by_kind: dict[FailureType, dict[str, list]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for path in paths:
        arm = path.name.split("-")[0]
        for n, line in enumerate(path.read_text().splitlines()):
            rec = json.loads(line)
            if rec["outcome"]["passed"]:
                continue
            for f in classify_record(rec).findings:
                by_kind[f.type][f.kind].append((f"{arm}{n:02d}", rec, f))
    picked, used = [], set()
    for t in (FailureType.TOOL, FailureType.ARG, FailureType.LOOP):
        pools = {k: rng.sample(v, len(v)) for k, v in sorted(by_kind[t].items())}
        got = 0
        while got < per_type and any(pools.values()):
            for kind in list(pools):  # round-robin over kinds, so rare kinds get checked too
                while pools[kind] and got < per_type:
                    conv, rec, f = pools[kind].pop()
                    if conv in used:
                        continue
                    used.add(conv)
                    picked.append({"id": f"{t.value}-{got + 1:02d}", "conv": conv, "rec": rec, "finding": f})
                    got += 1
                    break
    return picked


def show(item: dict) -> str:
    rec, f = item["rec"], item["finding"]
    view = agent_view_of(rec)
    before = [m for m in view[1 : f.turn] if m["role"] != "assistant"][-4:]
    lines = [
        f"### {item['id']}  {f.type.value}/{f.kind}  (conversation {item['conv']}, {rec['task_id'][:60]})",
        f"Finding: {f.tool or '-'}: {f.detail}",
        "Before the call:",
    ]
    lines += [f"  {m['role']}: {' '.join(str(m.get('content')).split())[:220]}" for m in before]
    m = view[f.turn]
    lines.append(f"Agent (turn {f.turn}): {' '.join(str(m.get('content') or '').split())[:200]}")
    results = {x.get("tool_call_id"): x.get("content") for x in view if x["role"] == "tool"}
    for tc in m.get("tool_calls") or []:
        fn = tc["function"]
        lines.append(
            f"  call {fn['name']}({str(fn['arguments'])[:120]}) -> {str(results.get(tc['id']))[:120]}"
        )
    if f.kind == "step_limit":
        lines.append(
            f"  conversation length {len(view)} messages; last agent turns repeat: "
            f"{len({str(x.get('content')) for x in view[-10:] if x['role'] == 'assistant'})} distinct of "
            f"{sum(x['role'] == 'assistant' for x in view[-10:])}"
        )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--per-type", type=int, default=10)
    p.add_argument("--labels", type=Path, help="JSON {sample id: correct?}; report precision")
    a = p.parse_args()
    if missing_files := [str(f) for f in a.files if not f.exists()]:
        print(f"⛔ not found: {', '.join(missing_files)}. Rollouts are local (results/ is gitignored):")
        print("   pass your own, e.g. `python scripts/spot_check_hallucination.py results/fixes/aya.jsonl`")
        return 1
    items = sample(a.files, a.per_type)
    if not a.labels:
        print("\n\n".join(show(x) for x in items))
        return 0
    labels = json.loads(a.labels.read_text())["labels"]
    missing = [x["id"] for x in items if x["id"] not in labels]
    if missing:
        print(f"⛔ no label for {', '.join(missing)}")
        return 1
    print("| Type | Correct | Precision | Kinds checked |\n|---|---:|---:|---|")
    for t in ("TOOL", "ARG", "LOOP"):
        mine = [x for x in items if x["id"].startswith(t)]
        ok = sum(bool(labels[x["id"]]) for x in mine)
        kinds = collections.Counter(x["finding"].kind for x in mine)
        print(
            f"| {t} | {ok} of {len(mine)} | {ok / len(mine):.0%} | "
            f"{', '.join(f'{k} {n}' for k, n in kinds.most_common())} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
