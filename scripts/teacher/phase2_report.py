"""Teacher bake-off phase 2: GLM-5.3-Flash vs Qwen3.8 27B (thinking off), paired (SEL-58).

    python scripts/teacher/phase2_report.py               # reads results/fixes/{glm53f,qwen38nt}.jsonl
    python scripts/teacher/phase2_report.py --plot docs/experiments/img/phase2_by_domain.png  # matplotlib

Attempt 1 only (temperature 0), judges on, on the train tasks both models finished. Paired exact
McNemar tests and paired bootstrap intervals per domain, plus failure patterns (how failed
conversations ended) and cost. Standard library only, except for --plot.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import statistics as st

_spec = importlib.util.spec_from_file_location(
    "bakeoff_report", pathlib.Path(__file__).parent / "bakeoff_report.py")
br = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(br)

DOMAINS = ("retail", "airline", "telecom")
MODELS = {"glm53f": "GLM-5.3-Flash", "qwen38nt": "Qwen3.8 27B (thinking off)"}


def load(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    """Attempt-1 records by (domain, task), skipping broken runs."""
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("attempt") == 1 and not r.get("error"):
            out[(r["domain"], r["task_id"])] = r
    return out


def transferred(r: dict) -> bool:
    """The agent handed the customer to a human instead of solving the problem."""
    return any(t["function"]["name"] == "transfer_to_human_agents"
               for m in r["messages"] for t in (m.get("tool_calls") or []))


def summary(recs: dict) -> dict:
    fails = [r for r in recs.values() if not r["outcome"]["passed"]]
    calls = sum(r["agent_usage"].get("calls", 0) for r in recs.values()) or 1
    return {
        "n": len(recs), "passed": sum(r["outcome"]["passed"] for r in recs.values()),
        "fail_max_steps": sum(r["termination"] == "max_steps" for r in fails),
        "fail_transferred": sum(transferred(r) for r in fails),
        "out_tokens_per_turn": sum(r["agent_usage"].get("completion_tokens", 0)
                                   for r in recs.values()) / calls,
        "agent_cost": sum(r["agent_usage"].get("est_cost") or 0 for r in recs.values()),
        "total_cost": sum((r["agent_usage"].get("est_cost") or 0) + (r.get("user_cost") or 0)
                          for r in recs.values()),
        "median_wall_s": st.median(r["wall_s"] for r in recs.values()) if recs else 0,
    }


def compare(a: dict, b: dict) -> dict:
    rows = {}
    for d in (*DOMAINS, "all"):
        keys = sorted(k for k in a.keys() & b.keys() if d in ("all", k[0]))
        pa = {k: a[k]["outcome"]["passed"] for k in keys}
        pb = {k: b[k]["outcome"]["passed"] for k in keys}
        only_a, only_b, p = br.mcnemar(pa, pb)
        rows[d] = {"shared": len(keys), "a": sum(pa.values()), "b": sum(pb.values()),
                   "only_a": only_a, "only_b": only_b, "p": p,
                   "diff_ci": br.paired_bootstrap(pa, pb) if keys else (0, 0),
                   "a_ci": br.wilson(sum(pa.values()), len(keys)),
                   "b_ci": br.wilson(sum(pb.values()), len(keys))}
    return rows


def plot(rows: dict, dst: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, aqua, ink, muted, grid, bg = "#2a78d6", "#1baf7a", "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
    doms = [*DOMAINS, "all"]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    w = 0.36
    for j, (key, label, c) in enumerate((("a", MODELS["glm53f"], blue), ("b", MODELS["qwen38nt"], aqua))):
        xs = [i + (j - 0.5) * (w + 0.03) for i in range(len(doms))]
        ys = [rows[d][key] / rows[d]["shared"] * 100 for d in doms]
        lo = [y - rows[d][f"{key}_ci"][0] * 100 for y, d in zip(ys, doms, strict=True)]
        hi = [rows[d][f"{key}_ci"][1] * 100 - y for y, d in zip(ys, doms, strict=True)]
        ax.bar(xs, ys, w, color=c, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=ink, alpha=0.5, lw=1, capsize=0, zorder=3)
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, 4, f"{y:.0f}%", ha="center", va="bottom", fontsize=8, color="white", zorder=4)
    ax.set_xticks(range(len(doms)))
    ax.set_xticklabels([f"{d}\n({rows[d]['shared']} tasks)" for d in doms], color=ink, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Passed first try, judges on (%; bars = 95% CI)", color=muted, fontsize=9)
    ax.set_title("Phase 2: same train tasks, paired", loc="left", color=ink, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", color=grid, lw=0.8, zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=muted, labelsize=8, length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=ink, bbox_to_anchor=(1, 1.12), ncol=2)
    fig.tight_layout()
    fig.savefig(dst, facecolor=bg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", type=pathlib.Path, default=pathlib.Path("results/fixes"))
    ap.add_argument("--plot")
    a = ap.parse_args()
    g, q = load(a.folder / "glm53f.jsonl"), load(a.folder / "qwen38nt.jsonl")
    rows = compare(g, q)
    print(f"{'domain':<8} {'tasks':>5} {'GLM':>13} {'Qwen-nt':>13} {'only GLM':>9} {'only Qwen':>10} "
          f"{'p':>9} {'GLM - Qwen (95%)':>18}")
    for d, r in rows.items():
        lo, hi = r["diff_ci"]
        print(f"{d:<8} {r['shared']:>5} {r['a']:>4} ({r['a'] / r['shared']:>4.0%})  {r['b']:>4} "
              f"({r['b'] / r['shared']:>4.0%}) {r['only_a']:>9} {r['only_b']:>10} {r['p']:>9.2g} "
              f"{lo:>+7.0%} to {hi:+.0%}")
    for name, recs in (("glm53f", g), ("qwen38nt", q)):
        for d in DOMAINS:
            s = summary({k: v for k, v in recs.items() if k[0] == d})
            print(f"{name:<9} {d:<8} {s['passed']:>3}/{s['n']:<3} fails: {s['fail_max_steps']} hit max "
                  f"steps, {s['fail_transferred']} handed to a human; {s['out_tokens_per_turn']:.0f} output "
                  f"tokens/turn; teacher ${s['agent_cost']:.2f}, total ${s['total_cost']:.2f}; "
                  f"median {s['median_wall_s']:.0f}s")
    if a.plot:
        plot(rows, a.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
