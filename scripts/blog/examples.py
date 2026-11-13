"""The blog's example graphics, in the figures' style (SEL-77).

    just blog-figures          # or: uv run --no-project --with matplotlib python scripts/blog/examples.py

Writes three images to docs/blog/img/:
- failure-types.png: the classifier's three failure types (Tool, Arg, Loop), each with a real example;
- reproduce-output.png: `just reproduce 2` as a terminal window, drawn from the script's own output
  (experiment 2's result rows and verdict are recomputed here, so the picture can't drift from them);
- test-table.png: Result 1's table (arms A, C and B on the test split), from the committed comparisons.
  Its alt text in the post carries the same numbers in words; `just blog-check` checks them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reproduce  # noqa: E402
from style import FONTS, HERO, INK, LEXSI, MONOS, MUTED, PAGE, PILL, caps, card  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs" / "blog" / "img"

FAILURES = [  # (label, title, what it means, example, how the example is drawn)
    (
        "Tool",
        "Calls a tool it doesn't have",
        "e.g. a phone tool only the\ncustomer can run",
        "check_apn_settings()",
        "one",
    ),
    (
        "Arg",
        "Invents an argument",
        "placeholder, made-up or\nill-typed values",
        'email="something@example.com"',
        "one",
    ),
    (
        "Loop",
        "Repeats itself",
        "the same failed call, the\nsame call or the same reply",
        "toggle_airplane_mode()",
        "stack",
    ),
]


def canvas(width: float, height: float) -> tuple[Figure, Axes]:
    """A figure in units of 10 px at 1×, drawn at 2× for crisp text."""
    fig = plt.figure(figsize=(width / 10, height / 10), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), width, height, facecolor=PAGE, edgecolor="none", zorder=0))
    return fig, ax


def tag(ax: Axes, right: float, y: float, text: str, z: float = 6) -> None:
    ax.text(
        right,
        y,
        text,
        ha="right",
        va="center",
        fontsize=9.5,
        family=FONTS,
        color="white",
        fontweight="bold",
        zorder=z,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.6", "facecolor": LEXSI, "edgecolor": "none"},
    )


def pill(ax: Axes, x: float, y: float, w: float, text: str, z: float = 3, fill: str = PILL) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            4.4,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=fill,
            edgecolor="none",
            zorder=z,
        )
    )
    ax.text(x + 1.4, y + 2.2, text, fontsize=10, family=MONOS, color=INK, va="center", zorder=z + 0.5)


def failure_types() -> Path:
    width, height = 128, 40
    fig, ax = canvas(width, height)
    ax.text(
        6,
        height - 4.8,
        caps("Three ways an agent hallucinates"),
        fontsize=10.5,
        family=FONTS,
        color=MUTED,
        fontweight="bold",
        va="center",
    )
    ax.text(
        width - 6,
        height - 4.8,
        "the classifier labels the first failing turn",
        fontsize=10.5,
        family=FONTS,
        color=MUTED,
        ha="right",
        va="center",
    )
    w, gap, y, h = 36, 4, 5, 25
    for k, (label, title, meaning, example, how) in enumerate(FAILURES):
        x = 6 + k * (w + gap)
        card(ax, x, y, w, h)
        ax.text(
            x + 3,
            y + h - 3.6,
            caps(f"Failure type {k + 1}"),
            fontsize=9.5,
            family=FONTS,
            color=MUTED,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.text(
            x + 3,
            y + h - 7.6,
            title,
            fontsize=15,
            family=FONTS,
            color=INK,
            fontweight="bold",
            va="center",
            zorder=4,
        )
        ax.text(
            x + 3,
            y + h - 10.4,
            meaning,
            fontsize=10.5,
            family=FONTS,
            color=MUTED,
            va="top",
            linespacing=1.4,
            zorder=4,
        )
        px, pw = x + 3, w - 6
        tag(ax, x + w - 3, y + h - 3.6, f"{label} ✗")
        if how == "stack":  # the same failing call, sent again and again
            for j, d in enumerate((1.6, 0.8, 0.0)):
                pill(
                    ax,
                    px + d,
                    y + 2.2 + d,
                    pw - 1.6,
                    example if j == 2 else "",
                    z=3 + j,
                    fill=("#e4e1d9", "#ebe8e1", PILL)[j],
                )
            ax.text(
                px + pw - 2.8,
                y + 4.4,
                "× 9",
                ha="right",
                va="center",
                fontsize=10,
                family=MONOS,
                color=MUTED,
                zorder=7,
            )
        else:
            pill(ax, px, y + 2.2, pw, example)
    return save(fig, IMG / "failure-types.png")


def reproduce_output() -> Path:
    """Experiment 2, as `just reproduce 2` prints it (question, result rows, verdict)."""
    e = reproduce.EXPERIMENTS[1]
    rows, verdict, _ = e.run()
    lines: list[tuple[str, str, bool]] = [  # (text, colour, bold)
        ("━━ Experiment 2 of 5: " + e.title, "white", True),
        ("", MUTED, False),
        ("Question: " + e.question, "#d6d3cb", False),
        ("", MUTED, False),
    ]
    for r in rows:
        lines.append(("  " + r, "#8c8a83" if set(r.strip()) <= {"─", " "} else "#e9e6df", False))
    lines += [("", MUTED, False)]
    words, line = ("Result: " + verdict).split(), ""
    for word in words:  # wrap the verdict at the window's width
        if len(line) + len(word) + 1 > 72:
            lines.append((line, "white", False))
            line = ""
        line = f"{line} {word}".strip()
    lines.append((line, "white", False))

    width, line_h = 100, 2.7
    win_h = 12 + line_h * len(lines) + 0.5
    height = win_h + 8
    fig, ax = canvas(width, height)
    x, y, w = 6, 4, width - 12
    card(ax, x, y, w, win_h, fill=HERO, radius=2.2, shadow=1.6)
    bar = y + win_h - 4.4
    for k in range(3):
        ax.add_patch(plt.Circle((x + 4 + k * 2.6, bar), 0.75, color="#3a3d45", zorder=4, linewidth=0))
    ax.text(
        x + 13, bar, "$ just reproduce 2", fontsize=10.5, family=MONOS, color="#a9a6a0", va="center", zorder=4
    )
    ax.plot([x, x + w], [bar - 3.2, bar - 3.2], color="#2c2f36", linewidth=1.0, zorder=4)
    ty = bar - 7.4
    for text, colour, bold in lines:
        if text.startswith("Result: "):
            ax.text(
                x + 4,
                ty,
                "Result:",
                fontsize=10,
                family=MONOS,
                color=LEXSI,
                fontweight="bold",
                va="center",
                zorder=4,
            )
            text, xo = text[len("Result:") :], 5.6
        else:
            xo = 0.0
        ax.text(
            x + 4 + xo,
            ty,
            text,
            fontsize=10,
            family=MONOS,
            color=colour,
            fontweight="bold" if bold else "normal",
            va="center",
            zorder=4,
        )
        ty -= line_h
    return save(fig, IMG / "reproduce-output.png")


TEST_ROWS = [
    ("success_rate", "Success"),
    ("hallucination_rate", "Conversations with a Tool or Arg failure"),
    ("loop_rate", "Conversations with a loop"),
]


def test_table_rows() -> list[tuple[str, float, float, float]]:
    """(label, Arm A, Arm C, Arm B) on the test split, from the paired comparisons."""
    ab, ac = reproduce.TEST_AB, reproduce.TEST_AC
    out = []
    for metric, label in TEST_ROWS:
        r_ab, r_ac = reproduce.cn.row(ab, metric), reproduce.cn.row(ac, metric)
        out.append((label, r_ab["a"], r_ac["b"], r_ab["b"]))
    return out


def test_table_alt() -> str:
    """The table in words, for the image's alt text (and `just blog-check`)."""
    return "; ".join(
        f"{label}: Arm A {a:.1%}, Arm C {c:.1%}, Arm B {b:.1%}" for label, a, c, b in test_table_rows()
    )


def test_table() -> Path:
    rows = test_table_rows()
    width, row_h = 128, 5.6
    h = 9 + row_h * len(rows) + 2
    height = h + 10
    fig, ax = canvas(width, height)
    ax.text(
        6,
        height - 4.4,
        caps("Test split · 100 tasks × 2 samples"),
        fontsize=10.5,
        family=FONTS,
        color=MUTED,
        fontweight="bold",
        va="center",
    )
    x, y, w = 6, 3, width - 12
    card(ax, x, y, w, h)
    cols = [x + w - 36, x + w - 22, x + w - 8]  # right edges of the three value columns
    # Arm B's column: a soft orange band, so the healed model reads first
    ax.add_patch(
        FancyBboxPatch(
            (cols[2] - 11.5, y + 1.6),
            14,
            h - 3.2,
            boxstyle="round,pad=0,rounding_size=1.4",
            facecolor="#fff1ea",
            edgecolor="none",
            zorder=3,
        )
    )
    head = y + h - 4.6
    ax.text(
        x + 4,
        head,
        caps("Measure"),
        fontsize=9.5,
        family=FONTS,
        color=MUTED,
        fontweight="bold",
        va="center",
        zorder=4,
    )
    for c, name, colour in zip(cols, ("Arm A", "Arm C", "Arm B"), (MUTED, MUTED, LEXSI), strict=True):
        ax.text(
            c,
            head,
            caps(name),
            fontsize=9.5,
            family=FONTS,
            color=colour,
            fontweight="bold",
            ha="right",
            va="center",
            zorder=4,
        )
    ax.plot([x + 4, x + w - 4], [head - 2.6, head - 2.6], color="#e6e3dc", linewidth=1.0, zorder=4)
    ty = head - 2.6 - row_h / 2
    for k, (label, a, c, b) in enumerate(rows):
        ax.text(x + 4, ty, label, fontsize=12, family=FONTS, color=INK, va="center", zorder=4)
        for col, v, strong in zip(cols, (a, c, b), (False, False, True), strict=True):
            ax.text(
                col,
                ty,
                f"{v:.1%}",
                fontsize=12.5 if strong else 12,
                family=FONTS,
                color=INK,
                ha="right",
                va="center",
                fontweight="bold" if strong else "normal",
                zorder=4,
            )
        if k < len(rows) - 1:
            ax.plot(
                [x + 4, x + w - 4], [ty - row_h / 2, ty - row_h / 2], color="#f0ede7", linewidth=0.8, zorder=4
            )
        ty -= row_h
    return save(fig, IMG / "test-table.png")


def save(fig: Figure, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=PAGE)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return out


def main() -> int:
    plt.rcParams["font.family"] = FONTS
    reproduce.BOLD = reproduce.DIM = reproduce.GREEN = reproduce.RED = reproduce.RESET = (
        ""  # plain text for the picture
    )
    failure_types()
    reproduce_output()
    test_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
