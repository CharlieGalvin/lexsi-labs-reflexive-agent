"""The blog's test-results chart (SEL-77): arms A, C and B on the τ²-bench test split.

    just blog-figures          # → docs/blog/img/test-results.png (and the loop diagram)

Left: task success by domain (mean of 2 samples over 100 test tasks, 95% bootstrap ranges over
tasks). Right: the share of conversations with a Tool or Arg hallucination (the classifier).
Two panels, not two axes: they are different measures. Every number is read from the committed
metrics files: `results/arm-*-test.eval.metrics.json` and
`results/arm-*-test-vs-arm-S2-test.compare.metrics.json`.

Needs matplotlib, which isn't a project dependency: run it with any Python that has it.
Colours (scripts/blog/style.py): the two controls in muted greys, the healed arm in Lexsi orange
(checked: worst colour-blind ΔE 12.9, normal-vision ΔE 18.7; Arm A's light grey is low-contrast
on white, so every bar group carries a value label and the post has the table).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import CARD, FONTS, GREY_1, GREY_2, INK, LEXSI, MUTED, PAGE, caps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "blog" / "img" / "test-results.png"

# Arms in the order they're read: the two controls, then the healed model
ARMS = [
    ("A", "Arm A · untuned", GREY_1),
    ("C", "Arm C · “be careful” prompt", GREY_2),
    ("S2", "Arm B · healed (SFT on verified fixes)", LEXSI),
]
DOMAINS = [
    ("overall", "All 100 tasks"),
    ("retail", "Retail (40)"),
    ("airline", "Airline (20)"),
    ("telecom", "Telecom (40)"),
]
SURFACE, INK_2, GRID = CARD, MUTED, "#ece9e2"  # panels are white cards on the warm page


def success(arm: str) -> dict[str, tuple[float, float, float]]:
    """{domain: (rate, low, high)} from the arm's eval metrics."""
    m = json.loads((RESULTS / f"arm-{arm}-test.eval.metrics.json").read_text())
    out = {"overall": (m["overall"]["success_rate"], *m["overall"]["success_ci95"])}
    for d, _ in DOMAINS[1:]:
        v = m["by_domain"][f"{arm}-{d}-test"]
        out[d] = (v["success_rate"], *v["success_ci95"])
    return out


def hallucination() -> dict[str, tuple[float, float, float]]:
    """{arm: (rate, low, high)}: Tool or Arg, from the paired comparisons with S2."""
    out = {}
    for other in ("A", "C"):
        c = json.loads((RESULTS / f"arm-{other}-test-vs-arm-S2-test.compare.metrics.json").read_text())
        row = next(r for r in c["rows"] if r["metric"] == "hallucination_rate")
        out[other] = (row["a"], *row["a_ci95"])
        out["S2"] = (row["b"], *row["b_ci95"])
    return out


def rounded_bar(ax, x: float, width: float, height: float, colour: str, radius_px: float = 4) -> None:  # noqa: ANN001
    """A bar anchored to the baseline with a rounded top (the data end)."""
    fig = ax.figure
    fig.canvas.draw_idle()
    bbox = ax.get_window_extent()
    xr, yr = ax.get_xlim(), ax.get_ylim()
    px_x = bbox.width / (xr[1] - xr[0])  # pixels per data unit
    px_y = bbox.height / (yr[1] - yr[0])
    r = min(radius_px / px_x, width / 2)
    aspect = px_x / px_y
    if height <= 0:
        return
    top = FancyBboxPatch(
        (x, 0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={r}",
        mutation_aspect=aspect,
        facecolor=colour,
        edgecolor="none",
        zorder=3,
    )
    ax.add_patch(top)
    ax.add_patch(
        plt.Rectangle(
            (x, 0), width, min(height, r * aspect * 1.2), facecolor=colour, edgecolor="none", zorder=3
        )
    )


def style(ax, title: str) -> None:  # noqa: ANN001
    ax.set_facecolor("none")  # the white card behind the panel shows through
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", colors=INK_2, labelsize=9, length=0)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.12)  # headroom above 100% for the value labels
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])


def main() -> int:
    succ = {arm: success(arm) for arm, _, _ in ARMS}
    hall = hallucination()

    plt.rcParams["font.family"] = FONTS
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 5.2), dpi=200, gridspec_kw={"width_ratios": [3.2, 1.3]}
    )
    fig.patch.set_facecolor(PAGE)
    style(left, "Task success on the test split")
    style(right, "Conversations with a Tool\nor Arg hallucination")

    width, gap = 0.24, 0.02  # bar width and the 2px-equivalent gap between adjacent bars
    n = len(ARMS)
    offsets = [(i - (n - 1) / 2) * (width + gap) for i in range(n)]
    left.set_xlim(-0.55, len(DOMAINS) - 0.45)
    right.set_xlim(-0.6, n - 0.4)
    fig.canvas.draw()

    for di, (d, _) in enumerate(DOMAINS):
        for (arm, _, colour), off in zip(ARMS, offsets, strict=True):
            rate, lo, hi = succ[arm][d]
            x = di + off - width / 2
            rounded_bar(left, x, width, rate, colour)
            left.plot([di + off] * 2, [lo, hi], color=INK_2, linewidth=0.9, zorder=4, solid_capstyle="round")
            if d == "overall" or arm == "S2":  # selective labels: the headline group and the healed arm
                left.text(
                    di + off,
                    hi + 0.025,
                    f"{rate:.0%}" if rate * 100 % 1 == 0 else f"{rate * 100:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=INK,
                    zorder=5,
                )
    left.set_xticks(range(len(DOMAINS)))
    left.set_xticklabels([label for _, label in DOMAINS], fontsize=9.5, color=INK)

    for i, (arm, _, colour) in enumerate(ARMS):
        rate, lo, hi = hall[arm]
        rounded_bar(right, i - 0.6 / 2, 0.6, rate, colour)
        right.plot([i, i], [lo, hi], color=INK_2, linewidth=0.9, zorder=4)
        right.text(
            i,
            hi + 0.025,
            f"{rate:.0%}" if rate * 100 % 1 == 0 else f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=INK,
            zorder=5,
        )
    right.set_xticks(range(n))
    right.set_xticklabels(["Arm A", "Arm C", "Arm B"], fontsize=9.5, color=INK)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for _, _, c in ARMS]
    fig.legend(
        handles,
        [label for _, label, _ in ARMS],
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        labelcolor=INK,
        handlelength=1.1,
        handleheight=1.1,
        columnspacing=1.6,
    )
    fig.text(
        0.025,
        0.952,
        caps("Result 1 · the test split"),
        fontsize=9.5,
        color=INK_2,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.025,
        0.025,
        "τ²-bench test split, 100 tasks × 2 samples, Aya Expanse 32B. Success oracle-verified; "
        "lines show 95% bootstrap ranges over tasks. Hallucination: Chiara's Tool / Arg classifier.",
        fontsize=7.5,
        color=INK_2,
    )
    fig.subplots_adjust(left=0.075, right=0.97, top=0.74, bottom=0.16, wspace=0.24)
    aspect = fig.get_figwidth() / fig.get_figheight()
    for ax in (left, right):  # each panel sits on a white card with a soft shadow
        bb = ax.get_position()
        x0, y0, w = bb.x0 - 0.05, bb.y0 - 0.075, bb.width + 0.065
        h = 0.885 - y0  # both cards share a top edge, with room for a two-line title
        for k in range(4):
            g = 0.003 * (k + 1)
            fig.add_artist(
                FancyBboxPatch(
                    (x0 - g / aspect, y0 - g - 0.004),
                    w + 2 * g / aspect,
                    h + 2 * g,
                    boxstyle=f"round,pad=0,rounding_size={0.03 + g}",
                    mutation_aspect=aspect,
                    transform=fig.transFigure,
                    facecolor="black",
                    edgecolor="none",
                    alpha=0.02,
                    zorder=-2,
                )
            )
        fig.add_artist(
            FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle="round,pad=0,rounding_size=0.03",
                mutation_aspect=aspect,
                transform=fig.transFigure,
                facecolor=CARD,
                edgecolor="none",
                zorder=-1,
            )
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=PAGE)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
