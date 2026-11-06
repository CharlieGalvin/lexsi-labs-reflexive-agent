"""Record a starter-task eval as results/<run>.metrics.json + a docs/EXPERIMENTS.md row, and
compare two recorded runs (docs/experiments/MESSY_LOOKUP.md).

    python scripts/experiment_log.py record runs/base-messy/eval_eval.csv base --arm A
    python scripts/experiment_log.py record runs/messy-grpo/eval_eval.csv messy-grpo \\
        --adapter runs/messy-grpo
    python scripts/experiment_log.py compare base messy-grpo     # a markdown table to post

``just baseline`` and ``just evaluate`` call ``record`` for you, straight after the eval.

**Metrics** (the starter task is Messy-ID Lookup, eval split: 120 questions on 40 held-out
customers): ``task_success`` = share answered correctly, with a 95% Wilson interval;
``calls_per_success`` = tool calls ÷ correct answers; ``hallucination_rate`` = share of questions
with an Arg error (a ``customer_id`` that doesn't exist, or a real one of the wrong customer).

**The log row** fills an empty placeholder row with the same run name (the ``base`` row), or is
added at the end of the table. A metrics file is never overwritten: record under a new run name.
Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LOG = ROOT / "docs" / "EXPERIMENTS.md"
TYPE_ORDER = ("clean", "lower_dash", "spaced", "number_only", "spelled", "distractor")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _rate(rows: list[dict], key: str) -> float:
    return round(sum(float(r[key]) for r in rows) / len(rows), 4) if rows else 0.0


def _arg_error(r: dict) -> bool:
    return float(r["hallucinated_ids"]) > 0 or float(r["wrong_customer"]) > 0


def metrics(rows: list[dict], *, run: str, arm: str | None, config: str, model: str,
            adapter: str | None, commit: str, owner: str, eval_csv: str,
            training: dict | None = None) -> dict:
    """The metrics record for one eval (``rows``: the eval CSV, one dict per question)."""
    n = len(rows)
    passed = sum(1 for r in rows if float(r["correct"]) == 1.0)
    calls = sum(float(r["n_tool_calls"]) for r in rows)
    halluc = sum(1 for r in rows if _arg_error(r))
    by_type = {}
    for t in [t for t in TYPE_ORDER if any(r["level"] == t for r in rows)]:
        sub = [r for r in rows if r["level"] == t]
        by_type[t] = {
            "passed": sum(1 for r in sub if float(r["correct"]) == 1.0),
            "n": len(sub),
            "tool_ok": _rate(sub, "tool_ok"),
            "hallucination_rate": round(sum(1 for r in sub if _arg_error(r)) / len(sub), 4),
        }
    return {
        "run": run,
        "date": dt.date.today().isoformat(),
        "owner": owner,
        "commit": commit,
        "config": config,
        "arm": arm,
        "task": "messy_lookup (starter task, docs/experiments/MESSY_LOOKUP.md)",
        "student": model + (f" + LoRA adapter {adapter}" if adapter else " (untuned)"),
        "split": "eval: 40 held-out customers x 3 questions",
        "decoding": "greedy",
        "questions": n,
        "passed": passed,
        "task_success": round(passed / n, 4) if n else 0.0,
        "task_success_wilson95": list(wilson(passed, n)),
        "tool_ok": _rate(rows, "tool_ok"),
        "format": _rate(rows, "format"),
        "hallucination_rate": round(halluc / n, 4) if n else 0.0,
        "hallucinated": halluc,
        "hallucination_method": "Arg error: a get_customer ID that doesn't exist, or a real ID of "
                                "the wrong customer (reflexive.messy_lookup.score)",
        "tool_calls_per_question": round(calls / n, 3) if n else 0.0,
        "calls_per_success": round(calls / passed, 2) if passed else None,
        "by_type": by_type,
        "training": training,
        "records": f"{eval_csv} (not committed)",
    }


def log_row(m: dict, change: str, notes: str = "") -> str:
    """One docs/EXPERIMENTS.md table row for metrics record ``m``."""
    lo, hi = m["task_success_wilson95"]
    success = (f"{m['passed']}/{m['questions']} = {m['task_success']:.1%} "
               f"(95% {lo:.1%}–{hi:.1%})")
    halluc = f"{m['hallucination_rate']:.1%} ({m['hallucinated']}/{m['questions']}, Arg)"
    by_type = ", ".join(f"{t} {v['passed']}/{v['n']}" for t, v in m["by_type"].items())
    links = [f"[metrics](../results/{m['run']}.metrics.json)"]
    tr = m.get("training") or {}
    if tr.get("wandb_url"):
        links.append(f"[W&B]({tr['wandb_url']})")
    if tr.get("hub_repo"):
        links.append(f"[adapter](https://huggingface.co/{tr['hub_repo']})")
    note = "; ".join(x for x in (notes, f"Correct by type: {by_type}", " ".join(links)) if x)
    calls = m["calls_per_success"] if m["calls_per_success"] is not None else "–"
    cells = [m["date"], m["run"], m["owner"], f"`{m['config']}` @ {m['commit']}", change,
             success, str(calls), halluc, note]
    return "| " + " | ".join(c.replace("|", "/") for c in cells) + " |"


def add_to_log(text: str, run: str, row: str) -> str:
    """``text`` (EXPERIMENTS.md) with ``row`` filling ``run``'s empty placeholder row, or added
    after the table's last row."""
    lines = text.splitlines()
    table = [i for i, line in enumerate(lines) if line.startswith("|")]
    if not table:
        raise ValueError("no table in the experiment log")
    for i in table[2:]:
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        if len(cells) > 5 and cells[1] == run and not cells[5]:
            lines[i] = row
            return "\n".join(lines) + "\n"
    lines.insert(table[-1] + 1, row)
    return "\n".join(lines) + "\n"


def record(a: argparse.Namespace) -> int:
    out = RESULTS / f"{a.run}.metrics.json"
    if out.exists():
        print(f"{out} exists; metrics are never overwritten: pick a new run name", file=sys.stderr)
        return 1
    with open(a.eval_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    summary_path = pathlib.Path(a.eval_csv).with_suffix(".json")
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    training = None
    if a.adapter and (pathlib.Path(a.adapter) / "run_summary.json").exists():
        training = json.loads((pathlib.Path(a.adapter) / "run_summary.json").read_text())
        training.pop("trainer", None)
    m = metrics(
        rows, run=a.run, arm=a.arm, config=a.config, adapter=a.adapter,
        model=summary.get("model", "CohereLabs/tiny-aya-global"),
        commit=a.commit or summary.get("commit") or _git("rev-parse", "--short", "HEAD"),
        owner=a.owner or _git("config", "user.name"), eval_csv=str(a.eval_csv), training=training,
    )
    RESULTS.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=2) + "\n")
    if a.change:
        change = a.change
    elif a.adapter:
        steps = (training or {}).get("steps", "?")
        change = f"GRPO {steps} steps (LoRA) on the untuned student; same eval questions as base"
    else:
        change = f"untuned baseline{f' (Arm {a.arm})' if a.arm else ''}: Tiny Aya on the starter task"
    row = log_row(m, change, a.notes)
    LOG.write_text(add_to_log(LOG.read_text(), a.run, row))
    print(f"wrote {out.relative_to(ROOT)} and a row in {LOG.relative_to(ROOT)}:\n{row}")
    return 0


def _pct(x: float) -> str:
    return f"{x:.1%}"


def compare_table(a: dict, b: dict) -> str:
    """A markdown table of run ``a`` vs run ``b`` (overall, then correct by question type)."""
    def pts(x: float, y: float) -> str:
        return f"{(y - x) * 100:+.1f} pts"

    def calls(m: dict) -> str:
        return "–" if m["calls_per_success"] is None else str(m["calls_per_success"])

    lines = [f"| Metric | {a['run']} | {b['run']} | Change |", "|---|---:|---:|---:|"]
    for label, key in (("task_success", "task_success"), ("tool_ok", "tool_ok"),
                       ("hallucination rate (Arg)", "hallucination_rate"), ("format", "format")):
        lines.append(f"| {label} | {_pct(a[key])} | {_pct(b[key])} | {pts(a[key], b[key])} |")
    if a["calls_per_success"] and b["calls_per_success"]:
        change = f"{b['calls_per_success'] - a['calls_per_success']:+.2f}"
    else:
        change = "–"
    lines.append(f"| calls / success | {calls(a)} | {calls(b)} | {change} |")
    for t in [t for t in TYPE_ORDER if t in a["by_type"] and t in b["by_type"]]:
        x, y = a["by_type"][t], b["by_type"][t]
        px, py = x["passed"] / x["n"], y["passed"] / y["n"]
        lines.append(f"| correct: `{t}` | {x['passed']}/{x['n']} | {y['passed']}/{y['n']} "
                     f"| {pts(px, py)} |")
    return "\n".join(lines)


def compare(a: argparse.Namespace) -> int:
    runs = []
    for name in (a.run_a, a.run_b):
        path = RESULTS / f"{name}.metrics.json"
        if not path.exists():
            print(f"no {path.relative_to(ROOT)}: record that run first", file=sys.stderr)
            return 1
        runs.append(json.loads(path.read_text()))
    print(compare_table(*runs))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="eval CSV -> results/<run>.metrics.json + a log row")
    r.add_argument("eval_csv", type=pathlib.Path)
    r.add_argument("run", help="run name, e.g. base or messy-grpo")
    r.add_argument("--arm", help="experiment arm, e.g. A")
    r.add_argument("--adapter", help="the trained adapter folder (reads its run_summary.json)")
    r.add_argument("--config", default="configs/train/messy_grpo.yaml")
    r.add_argument("--commit", help="default: the eval's own record, else HEAD")
    r.add_argument("--owner", help="default: git user.name")
    r.add_argument("--change", help="the log's 'Change vs. previous' cell")
    r.add_argument("--notes", default="", help="extra notes for the log row")
    c = sub.add_parser("compare", help="markdown table of two recorded runs")
    c.add_argument("run_a")
    c.add_argument("run_b")
    a = ap.parse_args()
    return record(a) if a.cmd == "record" else compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
