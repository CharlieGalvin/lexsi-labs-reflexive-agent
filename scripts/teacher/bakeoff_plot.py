"""Quality-vs-cost chart for the teacher bake-off (SEL-58). Needs matplotlib.

    python scripts/teacher/bakeoff_report.py --json results/bakeoff_report.json
    python scripts/teacher/bakeoff_plot.py results/bakeoff_report.json \
        docs/experiments/img/bakeoff_cost_vs_pass.png
"""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3, validated (light)
STUDENTS = {"aya", "tiny"}  # the Aya models: shown as diamonds, each direct-labelled
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a63", "#e6e5e0", "#fcfcfb"
OFFSETS = {"glm53f": (-9, 7, "right"), "dsv4f": (6, 8, "left"), "mimo": (8, -13, "left"),
           "gemma4": (8, -12, "left"), "qwen38": (8, 3, "left"), "haiku": (-9, 6, "right"),
           "aya": (8, 8, "left"), "tiny": (8, -13, "left")}


def main(src: str, dst: str) -> None:
    rows = [s for s in json.load(open(src))["rows"] if s["model"] != "qwen" and s["n"]]
    fig, ax = plt.subplots(figsize=(9, 5.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in rows:
        x, (lo, hi) = s["projection"]["total"], s["total_range"]
        y, ylo, yhi = s["rate"] * 100, s["ci"][0] * 100, s["ci"][1] * 100
        aya = s["model"] in STUDENTS
        c, partial = (AQUA if aya else BLUE if s["open"] else ORANGE), s["n"] < 30
        ax.errorbar(x, y, xerr=[[x - lo], [hi - x]], yerr=[[y - ylo], [yhi - y]], fmt="none",
                    ecolor=c, alpha=0.3, lw=1.2, zorder=1)
        marker = "D" if aya else "o" if s["open"] else "s"
        ax.scatter(x, y, s=80 if aya else 70, marker=marker, facecolor="none" if partial else c,
                   edgecolor=c if partial else BG, linewidth=2 if partial else 1.5, zorder=3)
        label = s["label"] + (f" ({s['n']} tasks)" if partial else "")
        label += ", reasoning likely off" if s["model"] == "gpt51" else ""
        label += " (cost if used as teacher)" if s["model"] == "aya" else ""
        label += " (customer cost only)" if s["model"] == "tiny" else ""
        dx, dy, ha = OFFSETS.get(s["model"], (8, -4, "left"))
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=INK, ha=ha)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 50, 100, 200, 500])
    ax.set_xticklabels(["$10", "$20", "$50", "$100", "$200", "$500"])
    ax.minorticks_off()
    ax.set_ylim(-9, 103)
    ax.set_xlabel("Projected teacher cost for the whole project (log scale; bars = sensitivity range)",
                  color=MUTED, fontsize=9)
    ax.set_ylabel("Passed, 30 train tasks (%; bars = 95% CI)", color=MUTED, fontsize=9)
    ax.set_title("Teacher bake-off: quality vs cost", loc="left", color=INK, fontsize=12, fontweight="bold")
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, label="Open weights", markersize=7),
        Line2D([], [], marker="s", ls="", color=ORANGE, label="Closed", markersize=7),
        Line2D([], [], marker="D", ls="", color=AQUA, label="Aya (student candidates)", markersize=6),
        Line2D([], [], marker="s", ls="", markerfacecolor="none", markeredgecolor=ORANGE,
               label="Unfinished run", markersize=7)],
        loc="center right", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(dst, facecolor=BG)


if __name__ == "__main__":
    main(*sys.argv[1:3])
