"""Build the benchmark hardness histogram."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import figstyle as S

ROOT = S.ROOT
NEIGHBORS = ROOT / "outputs" / "prep" / "identity_neighbors.csv"
BENCHMARK = ROOT / "outputs" / "prep" / "benchmark.json"
FEATURES = ROOT / "outputs" / "expanded_summary" / "expanded_mechanistic_group_features.csv"
OUT = S.FIG_DIR / "hardness_histogram.pdf"


def hardness_scores() -> dict[str, float]:
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    public_ids = {str(int(group["forget_id"])) for group in benchmark.get("public_groups", [])}
    public_ids.update(
        str(int(identity))
        for group in benchmark.get("public_groups", [])
        for identity in group.get("retain_ids", [])
    )
    dists = defaultdict(list)
    public_rows = 0
    for r in csv.DictReader(NEIGHBORS.open(encoding="utf-8", newline="")):
        if r["target_identity"] in public_ids or r["retain_identity"] in public_ids:
            public_rows += 1
        if int(r["rank"]) <= 3:
            dists[r["target_identity"]].append(float(r["cosine_distance"]))
    if public_rows:
        raise ValueError(
            "Saved neighbor graph contains public identities; rebuild it with the current "
            "split-first preparation pipeline before drawing this figure"
        )
    incomplete = {identity: len(values) for identity, values in dists.items() if len(values) != 3}
    if len(dists) != 5956 or incomplete:
        raise ValueError(
            "Hardness figure requires exactly three reranked non-public neighbors for all "
            f"5,956 identities; found {len(dists)} identities and {len(incomplete)} incomplete rows"
        )
    return {k: statistics.mean(v) for k, v in dists.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.apply(base_fontsize=15)

    hard = hardness_scores()
    forget_ids = [r["forget_id"] for r in csv.DictReader(FEATURES.open())]
    ranked_ids = sorted(hard, key=lambda identity: (hard[identity], int(identity)))
    if len(forget_ids) != 30 or set(forget_ids) != set(ranked_ids[:30]):
        raise ValueError("Figure cohort does not match the canonical benchmark's 30 hardest identities")
    selected = [hard[f] for f in forget_ids if f in hard]
    allvals = np.array(list(hard.values()))
    mu, sd = allvals.mean(), allvals.std()

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    fig.subplots_adjust(left=0.105, right=0.98, top=0.965, bottom=0.156)

    bins = np.linspace(allvals.min(), allvals.max(), 60)
    ax.hist(allvals, bins=bins, color=S.POLICY_COLOR["least_sim_hard"], alpha=0.35,
            edgecolor="white", linewidth=0.3, zorder=2)
    # Highlight bins containing selected identities.
    ax.hist(selected, bins=bins, color=S.POLICY_COLOR["nearest_hard"], alpha=0.95,
            edgecolor="white", linewidth=0.3, zorder=4)

    ax.axvline(mu, color=S.INK_MUTED, ls=(0, (5, 3)), lw=1.4, zorder=3)
    ax.text(mu + 0.004, ax.get_ylim()[1] * 0.94, f"pool mean {mu:.3f}", color=S.INK_MUTED,
            fontsize=12.5, ha="left", va="top")

    # Mark each selected identity below the histogram.
    selected_percent = 100.0 * len(selected) / len(allvals)
    ax.annotate(f"selected $n=30$ cohort\nhardest ${selected_percent:.1f}\\%$",
                xy=(np.mean(selected), 6), xytext=(0.47, 0.62), textcoords="axes fraction",
                fontsize=13.5, color=S.POLICY_COLOR["nearest_hard"],
                ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color=S.POLICY_COLOR["nearest_hard"], lw=1.8,
                                connectionstyle="arc3,rad=0.2"))

    ax.set_xlabel("mean ArcFace distance to three nearest identities (lower = harder)",
                  fontsize=13.5)
    ax.set_ylabel("number of identities")
    ax.grid(axis="x", visible=False)
    S.despine(ax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {args.out}  (pool={len(allvals)} selected={len(selected)} mean={mu:.3f} sd={sd:.3f})")


if __name__ == "__main__":
    main()
