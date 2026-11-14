"""The blog's self-healing loop, animated (SEL-77).

    just blog-figures          # or: uv run --no-project --with matplotlib python scripts/blog/loop_diagram.py

Writes docs/blog/img/heal-loop.gif (1280×720, loops forever) and a still, heal-loop.png. An orange
pulse travels round the loop; each stage lights up as it arrives and plays a real example from our
runs: the classifier flags an invented argument, the teacher rewrites the turn, the student trains
on the verified fixes, the gate compares dev success (11% → 60%, base model → SFT v2) and promotes
the new adapter, and the student's adapter ticks from v1 to v2.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from style import FAINT, FONTS, HERO, INK, LEXSI, LEXSI_SOFT, MONOS, MUTED, PAGE, PILL, caps, card  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_GIF = ROOT / "docs" / "blog" / "img" / "heal-loop.gif"
OUT_PNG = ROOT / "docs" / "blog" / "img" / "heal-loop.png"
FPS = 15
SIZE = (1280, 720)

# Canvas in units of 10 px: 128 × 72
W, H = 40, 18  # stage card
LX, RX, TY, BY = 6, 82, 44, 6  # card positions: left/right x, top/bottom y
SX, SY, SW, SH = 50, 25, 28, 18  # the student, in the middle

STAGES = [  # (x, y, label, title, body)
    (LX, TY, "01  Detect", "Find the failing turn", "A classifier labels it Tool, Arg or Loop."),
    (RX, TY, "02  Fix", "A teacher rewrites it", "It sees only what the student saw."),
    (RX, BY, "03  Retrain", "Learn from verified fixes", "QLoRA on turns the benchmark checked."),
    (LX, BY, "04  Gate", "Promote only if better", "The new adapter must solve more dev tasks."),
]
# Paths between the boxes, clockwise from the student: (polyline, label, label position)
PATHS: list[tuple[list[tuple[float, float]], str, tuple[float, float]]] = [
    ([(SX, 38), (LX + W / 2, 38), (LX + W / 2, TY)], "a conversation fails", (LX + W / 2 + 1.2, 38.6)),
    ([(LX + W, 53), (RX, 53)], "the failing turn", (64, 54.2)),
    ([(RX + W / 2, TY), (RX + W / 2, BY + H)], "verified fixes", (RX + W / 2, 34)),
    ([(RX, 15), (LX + W, 15)], "a candidate adapter", (64, 16.2)),
    ([(LX + W / 2, BY + H), (LX + W / 2, 30), (SX, 30)], "promote", (LX + W / 2 + 1.2, 30.6)),
]
ARG = 'email="something@example.com"'
FIX = '"Could you confirm your email?"'


@dataclass
class State:
    active: int | None = None  # the stage that's lit (0–3), or None
    travel: tuple[int, float] | None = None  # (path, 0–1): the pulse on its way
    pill: tuple[float, ...] = (0, 0, 0, 0)  # each stage's example, 0 (hidden) to 1 (done)
    version: int = 1  # the student's adapter
    glow: float = 0.0  # the student's flash after a promotion
    still: bool = False  # the static figure: every example shown, nothing lit


def point_on(path: list[tuple[float, float]], t: float) -> tuple[float, float]:
    segments = list(zip(path, path[1:], strict=False))
    lengths = [abs(b[0] - a[0]) + abs(b[1] - a[1]) for a, b in segments]
    d = t * sum(lengths)
    for (a, b), seg in zip(segments, lengths, strict=True):
        if d <= seg:
            f = d / seg if seg else 1.0
            return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f
        d -= seg
    return path[-1]


def draw_path(ax: Axes, i: int, s: State) -> None:
    pts, label, (lx, ly) = PATHS[i]
    lit = s.travel is not None and s.travel[0] == i
    colour = LEXSI if lit else ("#bdb9ae" if s.still else FAINT)
    xs, ys = zip(*pts, strict=True)
    ax.plot(xs, ys, color=colour, linewidth=2.2, solid_capstyle="round", solid_joinstyle="round", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            pts[-2],
            pts[-1],
            arrowstyle="-|>,head_length=7,head_width=4.5",
            color=colour,
            linewidth=0,
            zorder=1,
            shrinkA=0,
            shrinkB=0,
        )
    )
    ax.text(
        lx,
        ly,
        label,
        ha="center" if i in (1, 2, 3) else "left",
        va="center" if i == 2 else "bottom",
        fontsize=10.5,
        family=FONTS,
        color=LEXSI if lit else MUTED,
        style="italic",
        zorder=5,
        bbox={"facecolor": PAGE, "edgecolor": "none", "pad": 3} if i == 2 else None,
    )
    if lit and s.travel is not None:
        x, y = point_on(pts, s.travel[1])
        for r, alpha in [(1.7, 0.14), (1.1, 0.28)]:
            ax.add_patch(plt.Circle((x, y), r, color=LEXSI, alpha=alpha, zorder=6, linewidth=0))
        ax.add_patch(plt.Circle((x, y), 0.65, color=LEXSI, zorder=7, linewidth=0))


def pill_box(ax: Axes, x: float, y: float, w: float, fill: str = PILL, z: float = 3) -> None:
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


def type_out(text: str, p: float) -> str:
    return text[: round(len(text) * min(p / 0.7, 1.0))]


def tag(ax: Axes, right: float, y: float, text: str) -> None:
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
        zorder=5,
        bbox={"boxstyle": "round,pad=0.35,rounding_size=0.6", "facecolor": LEXSI, "edgecolor": "none"},
    )


def draw_stage(ax: Axes, i: int, s: State) -> None:
    x, y, label, title, body = STAGES[i]
    lit = s.active == i
    card(ax, x, y, W, H, edge=LEXSI if lit else "none", lw=2.2 if lit else 0, shadow=2.2 if lit else 1.0)
    ax.text(
        x + 3,
        y + H - 3.6,
        caps(label),
        fontsize=9.5,
        family=FONTS,
        color=LEXSI if lit or s.still else MUTED,
        fontweight="bold",
        va="center",
        zorder=4,
    )
    ax.text(
        x + 3,
        y + H - 7.6,
        title,
        fontsize=16,
        family=FONTS,
        color=INK,
        fontweight="bold",
        va="center",
        zorder=4,
    )
    ax.text(x + 3, y + H - 11.0, body, fontsize=10.5, family=FONTS, color=MUTED, va="center", zorder=4)
    p = 1.0 if s.still else s.pill[i]
    px, py, pw = x + 3, y + 1.8, W - 6
    pill_box(ax, px, py, pw)
    if p <= 0:
        return
    mono = {"fontsize": 10, "family": MONOS, "color": INK, "va": "center", "zorder": 4}
    if i == 0:  # the classifier flags an invented argument
        ax.text(px + 1.4, py + 2.2, type_out(ARG, p), **mono)
        if p > 0.75:
            tag(ax, px + pw - 1.0, py + 2.2, "Arg ✗")
    elif i == 1:  # the teacher's rewrite
        ax.text(px + 1.4, py + 2.2, type_out(FIX, p), **mono)
        if p > 0.75:
            tag(ax, px + pw - 1.0, py + 2.2, "✓")
    elif i == 2:  # training on the fixes
        f = min(p / 0.85, 1.0)
        pill_box(ax, px, py, max(pw * f, 2.4), fill=LEXSI_SOFT, z=3.5)
        ax.text(px + 1.4, py + 2.2, f"training on {round(146 * f)} verified conversations", **mono)
    else:  # the gate: dev success, base model against the new adapter
        new = 11 + round((60 - 11) * min(p / 0.7, 1.0))
        ax.text(px + 1.4, py + 2.2, f"dev success  11% → {new}%", **mono)
        if p > 0.75:
            tag(ax, px + pw - 1.0, py + 2.2, "promote ✓")


def draw_student(ax: Axes, s: State) -> None:
    for g in range(4) if s.glow > 0 else ():
        grow = 0.7 * (g + 1)
        ax.add_patch(
            FancyBboxPatch(
                (SX - grow, SY - grow),
                SW + 2 * grow,
                SH + 2 * grow,
                boxstyle=f"round,pad=0,rounding_size={2.2 + grow}",
                facecolor=LEXSI,
                edgecolor="none",
                alpha=0.08 * s.glow,
                zorder=1.5,
            )
        )
    card(ax, SX, SY, SW, SH, fill=HERO, radius=2.2, shadow=1.2)
    cx = SX + SW / 2
    ax.text(
        cx,
        SY + SH - 4.2,
        caps("Student agent"),
        ha="center",
        va="center",
        fontsize=9.5,
        family=FONTS,
        color="#9c998f",
        fontweight="bold",
        zorder=4,
    )
    ax.text(
        cx,
        SY + SH - 8.6,
        "Aya Expanse 32B",
        ha="center",
        va="center",
        fontsize=17,
        family=FONTS,
        color="white",
        fontweight="bold",
        zorder=4,
    )
    v2 = s.version == 2
    ax.text(
        cx,
        SY + 4.4,
        "+ LoRA adapter" if s.still else f"LoRA adapter v{s.version}",
        ha="center",
        va="center",
        fontsize=10.5,
        family=MONOS,
        zorder=4,
        color="white" if v2 or s.still else "#d6d3cb",
        bbox={
            "boxstyle": "round,pad=0.5,rounding_size=0.8",
            "facecolor": LEXSI if v2 or s.still else "#2c2f36",
            "edgecolor": "none",
        },
    )


def render(s: State) -> Image.Image:
    fig = plt.figure(figsize=(12.8, 7.2), dpi=200)
    fig.patch.set_facecolor(PAGE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 128)
    ax.set_ylim(0, 72)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 128, 72, facecolor=PAGE, edgecolor="none", zorder=0))
    ax.text(
        LX,
        67.2,
        caps("The self-healing loop"),
        fontsize=10.5,
        family=FONTS,
        color=MUTED,
        fontweight="bold",
        va="center",
    )
    ax.text(
        RX + W,
        67.2,
        "Reflexive Agents  ·  τ²-bench",
        fontsize=10.5,
        family=FONTS,
        color=MUTED,
        ha="right",
        va="center",
    )
    for i in range(len(PATHS)):
        draw_path(ax, i, s)
    for i in range(len(STAGES)):
        draw_stage(ax, i, s)
    draw_student(ax, s)
    fig.canvas.draw()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return img.resize(SIZE, Image.Resampling.LANCZOS)


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def timeline() -> list[State]:
    """The story, frame by frame: travel to a stage, play its example, hold; at the end, promote."""
    frames: list[State] = []
    pill = [0.0, 0.0, 0.0, 0.0]
    version = 1

    def hold(seconds: float, active: int | None = None) -> None:
        n = round(seconds * FPS)
        frames.extend(State(pill=tuple(pill), version=version, active=active) for _ in range(n))

    def travel(path: int, seconds: float = 0.7) -> None:
        n = round(seconds * FPS)
        frames.extend(
            State(pill=tuple(pill), version=version, travel=(path, ease(k / (n - 1)))) for k in range(n)
        )

    hold(0.8)
    for i in range(4):
        travel(i)
        n = round(1.5 * FPS)
        for k in range(n):
            pill[i] = k / (n - 1)
            frames.append(State(pill=tuple(pill), version=version, active=i))
        hold(0.6, active=i)
    travel(4)
    version = 2
    n = round(1.4 * FPS)
    frames.extend(State(pill=tuple(pill), version=version, glow=1 - k / (n - 1)) for k in range(n))
    hold(1.2)
    return frames


def main() -> int:
    plt.rcParams["font.family"] = FONTS
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    render(State(still=True)).save(OUT_PNG)
    images: list[Image.Image] = []
    durations: list[int] = []
    last = b""
    for s in timeline():
        img = render(s)
        raw = img.tobytes()
        if raw == last:  # the same as the previous frame: hold that one longer instead
            durations[-1] += round(1000 / FPS)
            continue
        images.append(img)
        durations.append(round(1000 / FPS))
        last = raw
    # One palette for every frame, from a sample of them (lit borders, the pulse, the glow, the end)
    sample = images[:: max(len(images) // 12, 1)] + [images[-1]]
    sheet = Image.new("RGB", (SIZE[0], SIZE[1] * len(sample)))
    for k, im in enumerate(sample):
        sheet.paste(im, (0, SIZE[1] * k))
    palette = sheet.quantize(colors=224, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    size = OUT_GIF.stat().st_size / 1e6
    print(
        f"wrote {OUT_GIF.relative_to(ROOT)} ({len(frames)} frames, {size:.1f} MB)"
        f" and {OUT_PNG.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
