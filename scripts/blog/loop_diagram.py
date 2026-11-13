"""The blog's self-healing loop diagram (SEL-77).

    just blog-figures          # or: uv run --no-project --with matplotlib python scripts/blog/loop_diagram.py

Writes docs/blog/img/heal-loop.png: the student in the centre, and the four stages of the loop
around it (detect, fix, retrain, gate), each with a concrete example. Colours: the data-viz
reference palette's first four slots, used only as stage accents; all text is in ink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "blog" / "img" / "heal-loop.png"
SURFACE, INK, INK_2, CARD, EDGE, PILL = "#fcfcfb", "#0b0b0b", "#52514e", "#ffffff", "#d9d8d3", "#f1f0ec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

W, H = 34, 17  # stage card size (axes units: 0–100 wide, 1–58 high)
L, R, TOP, BOT = 3, 63, 39, 3  # card positions: left/right x, top/bottom y
STAGES = [
    # (x, y, accent, number, title, body, example)
    (
        L,
        TOP,
        BLUE,
        "1",
        "Detect & classify",
        "A classifier finds the first failing turn\nand labels it Tool, Arg or Loop.",
        "invented order ID → Arg",
    ),
    (
        R,
        TOP,
        ORANGE,
        "2",
        "Teacher fix",
        "A stronger model rewrites that turn,\nseeing only what the student saw.",
        "“What's your order ID?”",
    ),
    (
        R,
        BOT,
        AQUA,
        "3",
        "Retrain",
        "The student's LoRA adapter is trained\non the verified fixes.",
        "no new Tool, Arg or Loop",
    ),
    (
        L,
        BOT,
        YELLOW,
        "4",
        "Gate",
        "The new adapter replaces the old one\nonly if it solves more dev tasks.",
        "✓ promote   ✗ keep old",
    ),
]


def card(
    ax: Axes, x: float, y: float, w: float, h: float, edge: str = EDGE, fill: str = CARD, lw: float = 1.0
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.6",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=2,
        )
    )


def stage(
    ax: Axes, x: float, y: float, accent: str, number: str, title: str, body: str, example: str
) -> None:
    card(ax, x, y, W, H)
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.1,
            H,
            boxstyle="round,pad=0,rounding_size=0.55",
            facecolor=accent,
            edgecolor="none",
            zorder=3,
        )
    )  # the stage's accent strip
    ax.add_patch(plt.Circle((x + 4.3, y + H - 3.4), 2.0, facecolor=accent, edgecolor="none", zorder=3))
    ax.text(
        x + 4.3,
        y + H - 3.45,
        number,
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + 7.6,
        y + H - 3.4,
        title,
        ha="left",
        va="center",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        x + 3.0, y + H - 7.0, body, ha="left", va="top", fontsize=9.3, color=INK_2, linespacing=1.35, zorder=4
    )
    card(ax, x + 3.0, y + 1.6, W - 6.0, 3.6, edge="none", fill=PILL)
    ax.text(
        x + 4.2,
        y + 3.4,
        example,
        ha="left",
        va="center",
        fontsize=8.8,
        color=INK,
        family="DejaVu Sans Mono",
        zorder=4,
    )


def arrow(
    ax: Axes,
    a: tuple[float, float],
    b: tuple[float, float],
    label: str,
    lx: float,
    ly: float,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=6,head_width=3.5",
            color=INK_2,
            linewidth=1.6,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        lx,
        ly,
        label,
        ha="center",
        va="center",
        fontsize=8.6,
        color=INK_2,
        style="italic",
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.5},
        zorder=5,
    )


def main() -> int:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 58)
    ax.axis("off")

    for s in STAGES:
        stage(ax, *s)

    # The student, in the middle of the loop: it serves customers, and a monitor stops loops live
    sx, sy, sw, sh = 35, 21.5, 30, 15
    card(ax, sx, sy, sw, sh, edge=INK, lw=1.4)
    ax.text(
        50,
        sy + 11.4,
        "Student agent",
        ha="center",
        va="center",
        fontsize=13,
        color=INK,
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        50,
        sy + 7.9,
        "Aya Expanse 32B + LoRA adapter",
        ha="center",
        va="center",
        fontsize=9.3,
        color=INK_2,
        zorder=4,
    )
    ax.plot([sx + 3, sx + sw - 3], [sy + 5.6, sy + 5.6], color=EDGE, linewidth=0.8, zorder=4)
    ax.text(
        50,
        sy + 2.9,
        "serves customers: retail, airline, telecom\na runtime monitor stops loops live",
        ha="center",
        va="center",
        fontsize=8.2,
        color=INK_2,
        linespacing=1.35,
        zorder=4,
    )

    # The cycle, clockwise: student → 1 → 2 → 3 → 4 → student
    lx, rx = L + W / 2, R + W / 2  # centres of the left and right cards
    arrow(ax, (sx, sy + 10), (lx, TOP), "a conversation fails", lx, 32.5, rad=-0.25)
    arrow(ax, (L + W, TOP + 8.5), (R, TOP + 8.5), "the failing turn", 50, TOP + 10.4)
    arrow(ax, (rx, TOP), (rx, BOT + H), "verified fixes", rx, 29.5)
    arrow(ax, (R, BOT + 8.5), (L + W, BOT + 8.5), "a candidate adapter", 50, BOT + 10.4)
    arrow(ax, (lx, BOT + H), (sx, sy + 3.5), "promote if better", lx, 25.5, rad=-0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
