"""Build the embedding-space overview and example outputs (Fig. 1)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

import figstyle as S

ROOT = S.ROOT
OUT = S.FIG_DIR / "intro_teaser.pdf"
NEAR_IMG = ROOT / "outputs" / "expanded_4016" / "nearest_hard_gen" / "forget" / "4016" / "034114" / "seed_101.png"
LEAST_IMG = ROOT / "outputs" / "expanded_4016" / "least_sim_hard_gen" / "forget" / "4016" / "034114" / "seed_101.png"
RED, BLUE = S.POLICY_COLOR["nearest_hard"], S.POLICY_COLOR["least_sim_hard"]


def load_square(path: Path):
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    s = min(h, w)
    return img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.apply(base_fontsize=15)

    W, H = 10.5, 5.2
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_aspect("equal"); ax.axis("off")

    # Embedding-space diagram.
    ax.add_patch(FancyBboxPatch((0.35, 0.62), 5.75, 4.35,
                 boxstyle="round,pad=0.02,rounding_size=0.12", fc="#f7f9fc",
                 ec="#c9d3e0", lw=1.4, zorder=0))
    ax.text(0.62, 4.72, "ArcFace recognition space", fontsize=16.5, color=S.INK_MUTED,
            style="italic", va="center", zorder=1)

    F = (2.55, 2.75)
    Rzone = 1.48
    # Verification region around the forgotten identity.
    ax.add_patch(Circle(F, Rzone, fc=(0.89, 0.32, 0.29, 0.11), ec=RED, ls=(0, (4, 3)),
                        lw=1.5, zorder=1))
    ax.text(F[0] - 0.02, F[1] + Rzone - 0.16, "still verifies\nas forget", fontsize=14.5,
            color=RED, ha="center", va="top", zorder=2, linespacing=1.15)

    N = (3.78, 3.34)   # nearest target (inside zone)
    L = (4.62, 1.42)   # least-sim target (outside zone)
    ax.add_patch(FancyArrowPatch(F, N, arrowstyle="-|>", mutation_scale=20, lw=2.6,
                 color=RED, zorder=4, shrinkA=7, shrinkB=6))
    ax.add_patch(FancyArrowPatch(F, L, arrowstyle="-|>", mutation_scale=20, lw=2.6,
                 color=BLUE, zorder=4, shrinkA=7, shrinkB=6))

    # Intermediate policy targets.
    for p in [(4.30, 2.82), (4.55, 2.12)]:
        ax.scatter(*p, s=46, c=S.INK_MUTED, alpha=0.6, zorder=5, edgecolors="white", linewidths=1)
    ax.text(4.78, 2.50, "random,\nmedian", fontsize=15, color=S.INK_MUTED,
            va="center", ha="left", linespacing=1.1, zorder=5)

    ax.scatter(*F, s=180, c=S.INK, zorder=6, edgecolors="white", linewidths=1.5)
    ax.text(F[0] - 0.12, F[1] - 0.30, "forget\nidentity", fontsize=15.5, color=S.INK,
            ha="center", va="top", fontweight="bold", zorder=6, linespacing=1.1)
    ax.scatter(*N, s=130, c=RED, zorder=6, edgecolors="white", linewidths=1.5)
    ax.text(N[0] + 0.09, N[1] + 0.16, "nearest-hard", fontsize=16, color=RED,
            va="bottom", ha="left", fontweight="bold", zorder=6)
    ax.scatter(*L, s=130, c=BLUE, zorder=6, edgecolors="white", linewidths=1.5)
    ax.text(L[0] + 0.05, L[1] - 0.22, "least-sim-hard", fontsize=16, color=BLUE,
            va="top", ha="center", fontweight="bold", zorder=6)

    ax.text(4.25, 0.30,
            "the projection adapter redirects the forget embedding\nbefore Arc2Face generation",
            fontsize=16, color=S.INK_SECONDARY, ha="center", va="center",
            style="italic", linespacing=1.05)

    # Generated faces at the redirect endpoints.
    def face(img_path, y_center, border):
        x0, x1 = 7.55, 9.25
        y0, y1 = y_center - 0.85, y_center + 0.85
        ax.imshow(load_square(img_path), extent=[x0, x1, y0, y1], zorder=3, aspect="auto")
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=border, lw=2.8, zorder=5))
        return x0, x1, y0, y1

    def mark(x, y, kind, color):
        d = 0.11
        ax.add_patch(FancyBboxPatch((x - 0.18, y - 0.18), 0.36, 0.36,
                     boxstyle="round,pad=0.02,rounding_size=0.06", fc="white", ec=color,
                     lw=1.6, zorder=7))
        if kind == "cross":
            for a, b in [((-d, -d), (d, d)), ((-d, d), (d, -d))]:
                ax.plot([x + a[0], x + b[0]], [y + a[1], y + b[1]], color=color, lw=3, zorder=8, solid_capstyle="round")
        else:
            pts = [(-0.12, 0.0), (-0.03, -0.11), (0.13, 0.13)]
            ax.plot([x + p[0] for p in pts], [y + p[1] for p in pts], color=color, lw=3, zorder=8,
                    solid_capstyle="round", solid_joinstyle="round")

    def chip(x0, y0, text):
        ax.text(x0 + 0.12, y0 + 0.24, text, fontsize=14.5, color="white", ha="left", va="center",
                zorder=7, bbox=dict(boxstyle="round,pad=0.3", fc=(0, 0, 0, 0.66), ec="none"))

    nx0, nx1, ny0, ny1 = face(NEAR_IMG, 3.55, RED)
    mark(nx1 - 0.26, ny1 - 0.26, "cross", RED)
    chip(nx0, ny0, "forget-sim 0.82")
    ax.text((nx0 + nx1) / 2, ny1 + 0.24, "still verifies as forget", fontsize=15.5, color=RED,
            ha="center", va="center", fontweight="bold")

    lx0, lx1, ly0, ly1 = face(LEAST_IMG, 1.45, BLUE)
    mark(lx1 - 0.26, ly1 - 0.26, "check", S.GREEN_TEXT)
    chip(lx0, ly0, "forget-sim 0.08")
    ax.text((lx0 + lx1) / 2, ly0 - 0.24, "forgotten", fontsize=15.5, color=BLUE,
            ha="center", va="center", fontweight="bold")

    # Connect the endpoints to their generated faces.
    ax.add_patch(FancyArrowPatch(N, (nx0, 3.55), arrowstyle="-|>", mutation_scale=16, lw=1.8,
                 color=RED, ls=(0, (2, 2)), zorder=4, shrinkA=6, shrinkB=2))
    ax.add_patch(FancyArrowPatch(L, (lx0, 1.45), arrowstyle="-|>", mutation_scale=16, lw=1.8,
                 color=BLUE, ls=(0, (2, 2)), zorder=4, shrinkA=6, shrinkB=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
