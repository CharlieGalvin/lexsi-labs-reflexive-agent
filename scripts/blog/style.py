"""The blog figures' shared look (SEL-77): a warm, muted page with one colour pop, Lexsi orange.

Neutral surfaces carry the structure; orange marks what matters (the healed model, the active
step, the flow). Text is always ink or muted ink, never the accent.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

PAGE = "#f5f4f0"  # warm off-white background
CARD = "#ffffff"
HERO = "#15171c"  # near-black: the one dark element per figure
INK = "#15171c"
MUTED = "#76736b"  # secondary text, small-caps labels
FAINT = "#d8d5cd"  # idle arrows, rules
PILL = "#f3f1ec"
LEXSI = "#ff4b0a"  # Lexsi Labs' logo orange: the only accent
LEXSI_SOFT = "#ffe4d9"
GREY_1 = "#c9c5bb"  # control arms in charts: muted, so the healed arm pops
GREY_2 = "#8f8b81"

_FAMILIES = ["Inter", "Lato", "DejaVu Sans"]  # the first one installed wins
_INSTALLED = {x.name for x in font_manager.fontManager.ttflist}
FONT = next((f for f in _FAMILIES if f in _INSTALLED), "DejaVu Sans")
MONO = next(
    (f for f in ["JetBrains Mono", "Ubuntu Mono", "DejaVu Sans Mono"] if f in _INSTALLED), "DejaVu Sans Mono"
)
FONTS = [FONT, "DejaVu Sans"]  # with a fallback for symbols (✓ ✗ →) the brand font lacks
MONOS = [MONO, "DejaVu Sans Mono"]


def caps(text: str) -> str:
    """Small-caps label with a little tracking (matplotlib has no letter-spacing)."""
    return " ".join(text.upper())


def card(
    ax: Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str = CARD,
    edge: str = "none",
    lw: float = 0.0,
    radius: float = 1.8,
    shadow: float = 1.0,
    z: float = 2,
) -> None:
    """A rounded card with a soft drop shadow (stacked translucent layers)."""
    for i in range(5):
        grow = 0.25 * (i + 1)
        ax.add_patch(
            FancyBboxPatch(
                (x - grow, y - grow - 0.35 * shadow),
                w + 2 * grow,
                h + 2 * grow,
                boxstyle=f"round,pad=0,rounding_size={radius + grow}",
                facecolor="#000000",
                edgecolor="none",
                alpha=0.018 * shadow,
                zorder=z - 0.1,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=edge,
            linewidth=lw,
            zorder=z,
        )
    )
