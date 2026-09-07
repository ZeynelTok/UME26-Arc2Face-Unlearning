"""Build the target-similarity and leakage summary figure (Fig. 3)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle as S

ROOT = S.ROOT
FEATURES = ROOT / "outputs" / "expanded_summary" / "expanded_mechanistic_group_features.csv"
ANALYSIS = ROOT / "outputs" / "expanded_summary" / "expanded_mechanistic_analysis.json"
OUT = S.FIG_DIR / "expanded_mechanistic_summary.pdf"


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def expected_per_sample_path(forget_id: str, policy: str) -> Path:
    return ROOT / "outputs" / f"expanded_{int(forget_id)}" / f"eval_{policy}" / "per_sample_results.csv"


def compute_arrival(features: list[dict]) -> dict[str, float]:
    """Closed-set target-arrival % among detected forget generations, per policy."""
    target_of = {
        str(int(row["forget_id"])): {p: str(int(row[f"{p}_target_identity"])) for p in S.POLICY_ORDER}
        for row in features
    }
    arrival = {}
    missing = []
    for pol in S.POLICY_ORDER:
        hit = det = 0
        for g, policy_targets in target_of.items():
            p = expected_per_sample_path(g, pol)
            if not p.exists():
                missing.append(p)
                continue
            tgt = policy_targets[pol]
            for r in read_rows(p):
                if r["role"] != "forget" or r["detector_success"] != "True":
                    continue
                det += 1
                if r["predicted_identity"] == tgt:
                    hit += 1
        arrival[pol] = 100.0 * hit / det if det else 0.0
    if missing:
        preview = "\n".join(str(path) for path in missing[:12])
        suffix = "" if len(missing) <= 12 else f"\n... and {len(missing) - 12} more"
        raise FileNotFoundError(f"Missing canonical per-sample files for Fig. 3:\n{preview}{suffix}")
    return arrival


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    S.apply(base_fontsize=15)
    features = read_rows(FEATURES)
    analysis = json.loads(ANALYSIS.read_text())
    per = analysis["per_policy"]
    arrival = compute_arrival(features)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 4.7))
    fig.subplots_adjust(left=0.068, right=0.985, top=0.90, bottom=0.150, wspace=0.24)

    # Panel (a): per-group target similarity and forget accuracy.
    for pol in S.POLICY_ORDER:
        xs = [float(r[f"{pol}_target_similarity"]) for r in features]
        ys = [float(r[f"{pol}_FA"]) for r in features]
        axa.scatter(xs, ys, s=46, c=S.POLICY_COLOR[pol], alpha=0.55,
                    edgecolors="#3a3a3a", linewidths=0.4, zorder=3,
                    label=S.POLICY_LABEL[pol])

    # Policy means and their trend line.
    axm = [per[p]["target_similarity_mean"] for p in S.POLICY_ORDER]
    aym = [per[p]["fa_mean"] for p in S.POLICY_ORDER]
    axa.plot(axm, aym, color="#2a2a2a", lw=1.6, zorder=5)
    for p, x, y in zip(S.POLICY_ORDER, axm, aym):
        axa.scatter([x], [y], s=340, c=S.POLICY_COLOR[p], edgecolors="white",
                    linewidths=2.0, zorder=6)

    axa.axvline(S.THRESHOLD, color=S.GREEN_TEXT, ls=(0, (5, 3)), lw=1.6, zorder=2)
    axa.text(S.THRESHOLD + 0.016, 34, f"verify threshold {S.THRESHOLD:.4f}",
             color=S.GREEN_TEXT, fontsize=13, ha="left", va="center", rotation=90)

    axa.set_xlim(0.0, 1.0)
    axa.set_ylim(-4, 108)
    axa.set_xlabel("chosen target similarity to forget centroid")
    axa.set_ylabel("Forget Accuracy (%) ↓")
    axa.set_title("(a)", loc="left")
    S.despine(axa)

    leg = axa.legend(loc="upper left", frameon=True, handletextpad=0.3,
                     borderpad=0.5, labelspacing=0.35)
    leg.get_frame().set_edgecolor(S.AXIS)
    leg.get_frame().set_facecolor("white")

    # Panel (b): target arrival and Leak@8.
    x = list(range(4))
    arr = [arrival[p] for p in S.POLICY_ORDER]
    leak = [per[p]["identity_leak8_mean"] for p in S.POLICY_ORDER]

    axb.plot(x, arr, color=S.GOOD_BLUE, lw=2.2, marker="o", ms=9,
             mec="white", mew=1.5, zorder=5, label="Target arrival ↑")
    axb.plot(x, leak, color=S.BAD_RED, lw=2.2, marker="s", ms=8.5,
             mec="white", mew=1.5, zorder=5, label="Leak@8 ↓")

    for xi, (a, l) in enumerate(zip(arr, leak)):
        # Offset each label away from the other series.
        arr_dy, leak_dy = (10, -16) if a >= l else (-16, 10)
        axb.annotate(f"{a:.0f}", (xi, a), textcoords="offset points",
                     xytext=(0, arr_dy), ha="center", fontsize=13.5, color=S.GOOD_BLUE)
        axb.annotate(f"{l:.0f}", (xi, l), textcoords="offset points",
                     xytext=(0, leak_dy), ha="center", fontsize=13.5, color=S.BAD_RED)

    axb.set_xlim(-0.3, 3.3)
    axb.set_ylim(-8, 112)
    axb.set_xticks(x)
    axb.set_xticklabels([S.POLICY_LABEL[p].replace("-hard", "\nhard") for p in S.POLICY_ORDER],
                        fontsize=14)
    axb.set_ylabel("percent of forget generations (%)")
    axb.set_title("(b)", loc="left")
    S.despine(axb)
    leg2 = axb.legend(loc="center", bbox_to_anchor=(0.62, 0.5),
                      frameon=True, handletextpad=0.4)
    leg2.get_frame().set_edgecolor(S.AXIS)
    leg2.get_frame().set_facecolor("white")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
