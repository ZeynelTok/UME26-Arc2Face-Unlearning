"""Build the ArcFace and AdaFace comparison figure (Fig. 5)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

import figstyle as S

ROOT = S.ROOT
PER_RUN = ROOT / "outputs" / "adaface_verification" / "adaface_vs_arcface_per_run.csv"
SUMMARY = ROOT / "outputs" / "adaface_verification" / "adaface_vs_arcface_summary.csv"
DOSE = ROOT / "outputs" / "adaface_verification" / "adaface_vs_arcface_dose_response.json"
OUT = S.FIG_DIR / "second_recognizer.pdf"

ARC, ADA = "#e34948", "#2a78d6"          # ArcFace / AdaFace
ARC_LABEL, ADA_LABEL = "ArcFace (antelopev2)", "AdaFace (IR-50)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    S.apply(base_fontsize=15)

    per_run = list(csv.DictReader(PER_RUN.open()))
    summ = {(r["policy"], r["recognizer"]): r for r in csv.DictReader(SUMMARY.open())}
    dose = json.loads(DOSE.read_text())["selection_similarity_dose_response"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    fig.subplots_adjust(left=0.088, right=0.985, top=0.90, bottom=0.160, wspace=0.22)

    # Panel (a): mean forget accuracy by policy and recogniser.
    import numpy as np
    x = np.arange(len(S.POLICY_ORDER))
    w = 0.38
    arc_fa = [float(summ[(p, "arcface")]["mean_FA"]) for p in S.POLICY_ORDER]
    ada_fa = [float(summ[(p, "adaface")]["mean_FA"]) for p in S.POLICY_ORDER]
    b1 = axa.bar(x - w / 2, arc_fa, w, color=ARC, label=ARC_LABEL, edgecolor="white", linewidth=1)
    b2 = axa.bar(x + w / 2, ada_fa, w, color=ADA, label=ADA_LABEL, edgecolor="white", linewidth=1)
    for bars in (b1, b2):
        for rect in bars:
            axa.annotate(f"{rect.get_height():.2f}", (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                         textcoords="offset points", xytext=(0, 3), ha="center",
                         fontsize=13.5, color=S.INK_SECONDARY)
    axa.set_xticks(x)
    axa.set_xticklabels(
        [S.POLICY_LABEL[p].replace("-hard", "\nhard") for p in S.POLICY_ORDER],
        fontsize=14,
    )
    axa.set_ylabel("Mean Forget Accuracy (%) ↓")
    axa.set_ylim(0, max(arc_fa) * 1.16)
    axa.set_title("(a)", loc="left")
    axa.grid(axis="x", visible=False)
    S.despine(axa)
    leg = axa.legend(loc="upper right", frameon=True)
    leg.get_frame().set_edgecolor(S.AXIS); leg.get_frame().set_facecolor("white")

    # Panel (b): target similarity against forget accuracy.
    def pts(recog):
        xs = [float(r["selection_target_similarity"]) for r in per_run if r["recognizer"] == recog]
        ys = [float(r["FA"]) for r in per_run if r["recognizer"] == recog]
        return xs, ys

    def anchors(recog):
        ax_, ay_ = [], []
        for p in S.POLICY_ORDER:
            rows = [r for r in per_run if r["recognizer"] == recog and r["policy"] == p]
            ax_.append(sum(float(r["selection_target_similarity"]) for r in rows) / len(rows))
            ay_.append(sum(float(r["FA"]) for r in rows) / len(rows))
        return ax_, ay_

    for recog, color, label in [("arcface", ARC, ARC_LABEL), ("adaface", ADA, ADA_LABEL)]:
        xs, ys = pts(recog)
        axb.scatter(xs, ys, s=34, c=color, alpha=0.35, edgecolors="none", zorder=3)
        ax_, ay_ = anchors(recog)
        r = dose[recog]["pearson_r"]
        axb.plot(ax_, ay_, color=color, lw=2.2, marker="o", ms=8, mec="white", mew=1.3,
                 zorder=5, label=f"{label}   $r={r:.2f}$")

    axb.set_xlim(0.0, 1.0); axb.set_ylim(-4, 108)
    axb.set_xlabel("chosen target similarity to forget centroid")
    axb.set_ylabel("Forget Accuracy (%) ↓")
    axb.set_title("(b)", loc="left")
    S.despine(axb)
    leg2 = axb.legend(loc="upper left", frameon=True, handletextpad=0.4)
    leg2.get_frame().set_edgecolor(S.AXIS); leg2.get_frame().set_facecolor("white")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.04)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
