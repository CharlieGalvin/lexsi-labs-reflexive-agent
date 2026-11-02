"""Before/after table for the README and the demo.

    python scripts/compare.py results/base.metrics.json results/grpo-eff-v0.metrics.json

Prints a Markdown table; arrows show whether each change is an improvement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HIGHER_IS_BETTER = {"task_success", "accuracy_answerable", "correct_abstention_unanswerable",
                    "auditkit/tool_call_validity", "auditkit/tool_selection",
                    "auditkit/agent_loop_detection"}  # 1.0 = no loop detected
LOWER_IS_BETTER = {"hallucination_rate", "avg_tool_calls", "redundant_call_rate",
                   "failed_call_rate", "malformed_call_rate", "no_answer_rate",
                   "calls_per_success", "auditkit/redundant_tool_calls"}


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    args = ap.parse_args()
    a, b = (json.loads(Path(p).read_text()) for p in (args.before, args.after))

    print(f"| metric | {Path(args.before).stem} | {Path(args.after).stem} | Δ |")
    print("|---|---:|---:|:-:|")
    for k in list(a) + [k for k in b if k not in a]:
        va, vb = a.get(k), b.get(k)
        mark = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va == va and vb == vb:
            d = vb - va
            if k in HIGHER_IS_BETTER:
                mark = "✅" if d > 0 else ("⚠️" if d < 0 else "")
            elif k in LOWER_IS_BETTER:
                mark = "✅" if d < 0 else ("⚠️" if d > 0 else "")
            mark = f"{d:+.3g} {mark}".strip()
        print(f"| {k} | {fmt(va)} | {fmt(vb)} | {mark} |")


if __name__ == "__main__":
    main()
