"""Shared colours and Matplotlib settings for the paper figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "paper_assets" / "figures"

# Policy colours run from the most similar to the most separated target.
POLICY_COLOR = {
    "nearest_hard": "#e34948",   # red    - most similar target, leakiest
    "random_hard": "#eb6834",    # orange
    "median_hard": "#eda100",    # yellow/amber
    "least_sim_hard": "#2a78d6",  # blue   - least similar target, cleanest
}
POLICY_LABEL = {
    "nearest_hard": "nearest-hard",
    "random_hard": "random-hard",
    "median_hard": "median-hard",
    "least_sim_hard": "least-sim-hard",
}
# Left-to-right order on categorical axes.
POLICY_ORDER = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]

# Colours used when a panel compares two metrics rather than four policies.
GOOD_BLUE = "#2a78d6"   # arrival (higher is better)
BAD_RED = "#e34948"     # leakage / forget accuracy (lower is better)

# Neutral colours for axes, labels, and backgrounds.
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
GREEN_TEXT = "#006300"

THRESHOLD = 0.14650361239910126


def _pick_serif() -> str:
    """Prefer a Times-like serif to match the LNCS body font."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Times New Roman", "Nimbus Roman", "DejaVu Serif"):
        if name in installed:
            return name
    return "DejaVu Serif"


def apply(base_fontsize: int = 15) -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [_pick_serif()],
        "mathtext.fontset": "dejavuserif",
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 2,
        "axes.titleweight": "normal",
        "axes.labelsize": base_fontsize + 2,
        "axes.labelweight": "normal",
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "svg.fonttype": "none",
    })


def despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
