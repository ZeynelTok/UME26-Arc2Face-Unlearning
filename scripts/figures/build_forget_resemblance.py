"""Build the per-policy forget-resemblance distributions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import figstyle as S

ROOT = S.ROOT
SUMMARY = ROOT / "outputs" / "adaface_verification" / "adaface_vs_arcface_summary.csv"
FEATURES = ROOT / "outputs" / "expanded_summary" / "expanded_mechanistic_group_features.csv"
OUT = S.FIG_DIR / "forget_resemblance.pdf"


def feature_group_ids() -> list[str]:
    return [str(int(r["forget_id"])) for r in csv.DictReader(FEATURES.open(newline="", encoding="utf-8"))]


def forget_sims(policy: str, group_ids: list[str]) -> np.ndarray:
    vals = []
    missing = []
    for gid in group_ids:
        p = ROOT / "outputs" / f"expanded_{int(gid)}" / f"eval_{policy}" / "per_sample_results.csv"
        if not p.exists():
            missing.append(p)
            continue
        for r in csv.DictReader(p.open(newline="", encoding="utf-8")):
            if r["role"] == "forget" and r["detector_success"] == "True":
                vals.append(float(r["intended_similarity"]))
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
        raise FileNotFoundError(f"Missing canonical per-sample files for forget-resemblance figure:\n{preview}{suffix}")
    return np.array(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.apply(base_fontsize=15)

    group_ids = feature_group_ids()
    fa = {r["policy"]: float(r["mean_FA"]) for r in csv.DictReader(SUMMARY.open())
          if r["recognizer"] == "arcface"}

    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    fig.subplots_adjust(left=0.197, right=0.975, top=0.965, bottom=0.140)

    grid = np.linspace(0.0, 0.9, 300)
    order_top_to_bottom = S.POLICY_ORDER  # nearest-hard (the failure) at top
    row_gap = 1.0
    height = 1.7  # KDE peak height in row units (>row_gap -> ridgeline overlap)

    for i, pol in enumerate(order_top_to_bottom):
        base = (len(order_top_to_bottom) - 1 - i) * row_gap
        vals = forget_sims(pol, group_ids)
        kde = gaussian_kde(vals, bw_method=0.35)
        dens = kde(grid)
        dens = dens / dens.max() * height
        color = S.POLICY_COLOR[pol]
        ax.fill_between(grid, base, base + dens, color=color, alpha=0.75, lw=0, zorder=10 + i)
        ax.plot(grid, base + dens, color="white", lw=1.2, zorder=10 + i)
        # Median marker and label.
        med = float(np.median(vals))
        h_at = np.interp(med, grid, dens)
        ax.plot([med, med], [base, base + h_at], color="white", lw=1.6, zorder=11 + i)
        ax.plot([med, med], [base, base + h_at], color=S.INK, lw=1.0, ls=(0, (2, 1.5)), zorder=11 + i)
        ax.text(med, base + h_at + 0.10, f"median {med:.2f}", fontsize=13.5,
                color=S.INK, ha="center", va="bottom", zorder=30)
        # Policy name and mean forget accuracy.
        ax.text(-0.025, base + 0.55, S.POLICY_LABEL[pol], fontsize=14, fontweight="bold",
                color=color, ha="right", va="center", transform=ax.get_yaxis_transform())
        ax.text(-0.025, base + 0.22, f"FA {fa[pol]:.2f}%", fontsize=14,
                color=S.INK_MUTED, ha="right", va="center", transform=ax.get_yaxis_transform())

    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(-0.15, (len(order_top_to_bottom) - 1) * row_gap + height + 0.5)
    ax.set_yticks([])
    ax.set_xlabel("cosine similarity to forget identity (higher = greater forget resemblance)",
                  fontsize=15)
    ax.grid(axis="y", visible=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
