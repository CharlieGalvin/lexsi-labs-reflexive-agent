"""Teacher bake-off report: pass rates, paired statistics and projected cost (SEL-58).

    python scripts/teacher/bakeoff_report.py                     # reads results/episodes/sample-*.jsonl
    python scripts/teacher/bakeoff_report.py --json results/bakeoff_report.json

Standard library only, so it runs anywhere the results are. What it computes:

- **Pass rate with a 95% Wilson interval** per model (30 tasks each, database checks only).
- **Paired comparisons.** Every model saw the same 30 tasks, so models are compared task by task
  with an exact McNemar test (a two-sided binomial test on the tasks where exactly one of the
  two passed). Paired tests are more sensitive than comparing two separate proportions. P-values
  among the top tier are Holm-corrected for the number of comparisons.
- **Paired bootstrap** (10,000 resamples of the 30 tasks) for the difference in pass rate.
- **Minimum detectable difference** for 30 paired tasks, and the sample size phase 2 needs.
- **Projected cost of the teacher over the whole pipeline**, with a bootstrap range from the
  per-conversation cost and a sensitivity range over the assumptions (see ``ASSUMPTIONS``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from itertools import combinations
from typing import Any

from reflexive.stats import holm, mcnemar, mcnemar_power, paired_bootstrap, wilson

LABELS = {
    "qwen38": "Qwen3.8 27B",
    "sonnet": "Claude Sonnet 5.5",
    "glm53f": "GLM-5.3-Flash",
    "dsv4f": "DeepSeek V4 Flash",
    "mimo": "MiMo-V2.6-Flash",
    "gem38f": "Gemini 3.8 Flash",
    "gemma4": "Gemma 4 31B",
    "haiku": "Claude Haiku 5.5",
    "gptoss": "gpt-oss-120b",
    "glm": "GLM-4.7-Flash",
    "gpt51": "GPT-5.1",
    "opus": "Claude Opus 5.5",
    "qwen": "Qwen3 32B",
    "aya": "Aya Expanse 32B",
    "tiny": "Tiny Aya",
}
OPEN = {"qwen38", "glm53f", "dsv4f", "mimo", "gemma4", "gptoss", "glm", "qwen", "aya", "tiny"}
STUDENTS = {"aya", "tiny"}
SKIP = {"gflite"}  # dropped from the bake-off on 3 Nov

# How the teacher is used over the project (see docs/experiments/TEACHER_BAKEOFF_RESULTS.md)
ASSUMPTIONS = {
    "train_tasks": {"retail": 74, "airline": 30, "telecom": 74},  # τ²-bench v2.0 train split
    "student_fail_rate": 0.97,  # Aya Expanse 32B failed 29/30 of our sample
    "attempts": 3,  # best-of-3, stopping at the first verified pass
    "cycle_weights": [1.0, 0.6, 0.4],  # 3 heal cycles, fewer failures each time
    "telecom_cost_x": 2.0,  # a telecom conversation costs ~2x (Aya check: 182K vs ~90K tokens)
    "phase2": {"tasks": 60, "telecom_tasks": 10, "attempts": 3},
}
SENSITIVITY = {
    "telecom_cost_x": (1.5, 3.0),
    "student_fail_rate": (0.85, 1.0),
    "cycle_weights": ([1.0], [1.0, 0.8, 0.6, 0.4, 0.2]),
}


# --- Cost projection ----------------------------------------------------------------------------


def project(p: float, conv_cost: float, a: dict = ASSUMPTIONS) -> dict:
    tr = a["train_tasks"]
    rr = (tr["retail"] + tr["airline"]) * a["student_fail_rate"]
    tel = tr["telecom"] * a["student_fail_rate"]
    attempts = sum((1 - p) ** i for i in range(a["attempts"]))  # stop at the first pass
    yield_ = 1 - (1 - p) ** a["attempts"]
    cycle1 = (rr + tel * a["telecom_cost_x"]) * attempts * conv_cost
    p2 = a["phase2"]
    phase2 = (p2["tasks"] + p2["telecom_tasks"] * a["telecom_cost_x"]) * p2["attempts"] * conv_cost
    total = phase2 + cycle1 * sum(a["cycle_weights"])
    fixes = (rr + tel) * yield_
    return {
        "phase2": phase2,
        "cycle1": cycle1,
        "total": total,
        "fixes_cycle1": fixes,
        "per_fix": cycle1 / fixes if fixes else float("inf"),
        "attempts_per_task": attempts,
    }


def total_range(p: float, conv_cost: float) -> tuple[float, float]:
    """Lowest and highest projected total over the sensitivity ranges."""
    totals = []
    for tx in SENSITIVITY["telecom_cost_x"]:
        for fr in SENSITIVITY["student_fail_rate"]:
            for cw in SENSITIVITY["cycle_weights"]:
                a = {**ASSUMPTIONS, "telecom_cost_x": tx, "student_fail_rate": fr, "cycle_weights": cw}
                totals.append(project(p, conv_cost, a)["total"])
    return min(totals), max(totals)


# --- Loading ------------------------------------------------------------------------------------


def load(folder: pathlib.Path) -> dict[str, list[dict]]:
    out = {}
    for f in sorted(folder.glob("sample-*.jsonl")):
        name = f.stem.removeprefix("sample-")
        if name in SKIP or name not in LABELS:
            continue
        out[name] = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    return out


def summarise(name: str, recs: list[dict]) -> dict:
    n = len(recs)
    k = sum(r["outcome"]["passed"] for r in recs)
    agent = [r["agent_usage"].get("est_cost") or 0.0 for r in recs]
    user = [r.get("user_cost") or 0.0 for r in recs]
    conv = [a + u for a, u in zip(agent, user, strict=True)]
    rng = random.Random(2)
    boots = sorted(sum(rng.choice(conv) for _ in conv) / n for _ in range(2000)) if n else [0.0]
    lo, hi = wilson(k, n)
    p = k / n if n else 0.0
    proj = project(p, sum(conv) / n) if n else {}
    return {
        "model": name,
        "label": LABELS[name],
        "open": name in OPEN,
        "n": n,
        "passed": k,
        "rate": p,
        "ci": [lo, hi],
        "by_domain": {
            d: [
                sum(r["outcome"]["passed"] for r in recs if r["domain"] == d),
                sum(r["domain"] == d for r in recs),
            ]
            for d in ("retail", "airline")
        },
        "invented_ids": sum(bool(r["diagnostics"]["unseen_arguments"]) for r in recs),
        "broken": sum(bool(r.get("error")) for r in recs),
        "agent_per_conv": sum(agent) / n if n else 0.0,
        "user_per_conv": sum(user) / n if n else 0.0,
        "conv_cost_ci": [boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]],
        "mean_wall_s": sum(r.get("wall_s", 0) for r in recs) / n if n else 0.0,
        "billed": any(r["agent_usage"].get("billed_cost") for r in recs),
        "projection": proj,
        "total_range": list(total_range(p, sum(conv) / n)) if n else [0, 0],
        "passed_tasks": {f"{r['domain']}:{r['task_id']}": bool(r["outcome"]["passed"]) for r in recs},
    }


# --- Report -------------------------------------------------------------------------------------


def report(folder: pathlib.Path) -> dict:
    data = load(folder)
    rows = sorted((summarise(m, r) for m, r in data.items()), key=lambda s: (-s["rate"], s["model"]))
    full = [s for s in rows if s["n"] == 30 and s["model"] not in STUDENTS]
    best = max(full, key=lambda s: s["passed"])
    top = [s for s in full if s["passed"] / s["n"] >= best["ci"][0] and s["invented_ids"] <= 5]

    pairs: list[dict[str, Any]] = []
    for a, b in combinations(top, 2):
        oa, ob, p = mcnemar(a["passed_tasks"], b["passed_tasks"])
        pairs.append(
            {
                "a": a["label"],
                "b": b["label"],
                "only_a": oa,
                "only_b": ob,
                "p": p,
                "diff_ci": paired_bootstrap(a["passed_tasks"], b["passed_tasks"]),
            }
        )
    for pr, adj in zip(pairs, holm([pr["p"] for pr in pairs]), strict=True):
        pr["p_holm"] = adj

    vs_baseline = []
    base = {s["model"]: s for s in rows}
    for ref in ("aya", "glm"):
        if ref not in base:
            continue
        for s in full:
            if s["model"] == ref:
                continue
            oa, ob, p = mcnemar(s["passed_tasks"], base[ref]["passed_tasks"])
            vs_baseline.append(
                {"model": s["label"], "vs": base[ref]["label"], "only_model": oa, "only_ref": ob, "p": p}
            )

    task_fail = {}
    for s in full:
        for t, ok in s["passed_tasks"].items():
            task_fail.setdefault(t, 0)
            task_fail[t] += not ok
    hardest = sorted(task_fail.items(), key=lambda kv: -kv[1])[:6]

    power = {
        f"{a:.0%} vs {b:.0%}": {n: mcnemar_power(a, b, n) for n in (30, 70, 210)}
        for a, b in ((0.95, 0.85), (0.95, 0.75), (0.90, 0.60))
    }
    return {
        "rows": rows,
        "top_tier": [s["label"] for s in top],
        "top_pairs": pairs,
        "vs_baseline": vs_baseline,
        "hardest_tasks": hardest,
        "power": power,
        "assumptions": ASSUMPTIONS,
        "sensitivity": SENSITIVITY,
    }


def print_report(r: dict) -> None:
    print(
        f"{'model':<20} {'open':>4} {'passed':>7} {'95% CI':>11} {'inv':>4} {'$agent':>7} "
        f"{'$cust':>6} {'$/fix':>6} {'total':>7} {'range':>13}"
    )
    for s in r["rows"]:
        if not s["n"]:
            continue
        pr = s["projection"]
        print(
            f"{s['label'][:20]:<20} {'yes' if s['open'] else 'no':>4} {s['passed']:>3}/{s['n']:<3} "
            f"{s['ci'][0]:>4.0%}–{s['ci'][1]:<4.0%} {s['invented_ids']:>4} {s['agent_per_conv']:>7.4f} "
            f"{s['user_per_conv']:>6.4f} {pr['per_fix']:>6.3f} ${pr['total']:>6.0f} "
            f"${s['total_range'][0]:>4.0f}–{s['total_range'][1]:<5.0f}"
        )
    print(f"\nTop tier (within the best model's 95% CI, ≤5 invented-ID conversations): {r['top_tier']}")
    print("\nPaired McNemar tests within the top tier (Holm-corrected):")
    for pr in r["top_pairs"]:
        lo, hi = pr["diff_ci"]
        print(
            f"  {pr['a']} vs {pr['b']}: {pr['only_a']} vs {pr['only_b']} discordant tasks, "
            f"p = {pr['p']:.3f} (Holm {pr['p_holm']:.3f}), difference {lo:+.0%} to {hi:+.0%}"
        )
    print("\nAgainst the baselines:")
    for v in r["vs_baseline"]:
        print(f"  {v['model']} vs {v['vs']}: {v['only_model']} vs {v['only_ref']}, p = {v['p']:.2g}")
    print("\nHardest tasks (failed by how many full-run teachers):", r["hardest_tasks"])
    print("\nPower of a paired McNemar test (alpha 0.05):")
    for k, v in r["power"].items():
        print(f"  {k}: " + ", ".join(f"n={n}: {p:.0%}" for n, p in v.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=pathlib.Path, default=pathlib.Path("results/episodes"))
    ap.add_argument("--json", type=pathlib.Path)
    a = ap.parse_args()
    r = report(a.folder)
    print_report(r)
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, default=list))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
