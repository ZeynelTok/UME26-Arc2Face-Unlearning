"""Re-score expanded generations with ArcFace and AdaFace."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.adaface_utils import AdaFaceEmbedder
from genmu_face_unlearning.benchmark import iter_groups, load_benchmark
from genmu_face_unlearning.evaluation import evaluate_run
from genmu_face_unlearning.threshold import load_threshold_calibration

POLICIES = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]


def _build_group_lookup(benchmark_paths: list[Path]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for path in benchmark_paths:
        if not path.exists():
            continue
        for group in iter_groups(load_benchmark(path)):
            lookup.setdefault(int(group["forget_id"]), group)
    return lookup


def _discover_matched_groups(out_root: Path, min_pngs: int = 168) -> list[str]:
    present: dict[str, set[str]] = {p: set() for p in POLICIES}
    for group_dir in sorted(out_root.glob("expanded_*")):
        gid = group_dir.name.replace("expanded_", "")
        for policy in POLICIES:
            gen_dir = group_dir / f"{policy}_gen"
            if gen_dir.exists() and len(list(gen_dir.glob("**/*.png"))) >= min_pngs:
                present[policy].add(gid)
    matched = present[POLICIES[0]]
    for policy in POLICIES[1:]:
        matched = matched & present[policy]
    return sorted(matched, key=int)


def _selection_target_similarity(out_root: Path, gid: str, policy: str) -> float:
    summary_path = out_root / f"expanded_{gid}" / f"u_projection_adapter_{policy}" / "projection_adapter_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["target_similarity_to_forget"])


def _score(embedder, threshold, group, config, gen_dir, det_thresh):
    results, metrics = evaluate_run(
        config=config,
        benchmark_path=None,
        threshold_path=None,
        after_dir=gen_dir,
        det_thresh=det_thresh,
        embedder=embedder,
        threshold=threshold,
        group=group,
    )
    return results, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--benchmarks",
        type=Path,
        nargs="+",
        default=[ROOT / "outputs" / "prep" / "benchmark.json"],
    )
    parser.add_argument("--arcface-threshold", type=Path, default=ROOT / "outputs" / "prep" / "threshold_arcface.json")
    parser.add_argument("--adaface-threshold", type=Path, default=ROOT / "outputs" / "prep" / "threshold_adaface.json")
    parser.add_argument("--insightface-model-root", type=Path, default=ROOT / "models" / "insightface")
    parser.add_argument("--adaface-weights", type=Path, default=ROOT / "models" / "adaface" / "weights" / "adaface_ir50_webface4m.ckpt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "adaface_verification")
    parser.add_argument(
        "--recognizers",
        nargs="+",
        choices=("arcface", "adaface"),
        default=["arcface", "adaface"],
        help="Recognizers to rescore. Existing rows for unselected recognizers are preserved.",
    )
    args = parser.parse_args()

    matched = _discover_matched_groups(args.out_root)
    if not matched:
        raise SystemExit("No groups found with all four policies present.")
    print(f"Matched groups (all 4 policies present): {len(matched)} -> {matched}")

    lookup = _build_group_lookup(args.benchmarks)
    missing = [gid for gid in matched if int(gid) not in lookup]
    if missing:
        raise SystemExit(f"Group configs not found in benchmarks for: {missing}")

    arc_threshold = load_threshold_calibration(args.arcface_threshold)
    ada_threshold = load_threshold_calibration(args.adaface_threshold)

    recognizers = []
    if "arcface" in args.recognizers:
        print("Loading ArcFace embedder...")
        arc_embedder = ArcFaceEmbedder(
            model_root=args.insightface_model_root,
            device=args.device,
            det_thresh=args.det_thresh,
        )
        recognizers.append(("arcface", arc_embedder, arc_threshold))
    if "adaface" in args.recognizers:
        print("Loading AdaFace embedder...")
        ada_embedder = AdaFaceEmbedder(
            insightface_model_root=args.insightface_model_root,
            adaface_weights=args.adaface_weights,
            device=args.device,
            det_thresh=args.det_thresh,
        )
        recognizers.append(("adaface", ada_embedder, ada_threshold))

    rows = []
    adaface_prediction_frames = []
    for gid in matched:
        group = lookup[int(gid)]
        config = {
            "forget_id": int(gid),
            "hard_retain_ids": [int(x) for x in group.get("hard_retain_ids", [])],
            "retain_ids": [int(x) for x in group.get("retain_ids", [])],
        }
        for policy in POLICIES:
            gen_dir = args.out_root / f"expanded_{gid}" / f"{policy}_gen"
            if not gen_dir.exists():
                continue
            selection_sim = _selection_target_similarity(args.out_root, gid, policy)
            for rec_name, embedder, threshold in recognizers:
                results, metrics = _score(
                    embedder, threshold, group, config, gen_dir, args.det_thresh
                )
                rows.append(
                    {
                        "group": int(gid),
                        "policy": policy,
                        "recognizer": rec_name,
                        "FA": metrics["FA"],
                        "RA": metrics["RA"],
                        "ERB": metrics["ERB"],
                        "IdentityLeak@8": metrics["IdentityLeak@8"],
                        "selection_target_similarity": selection_sim,
                        "mean_target_similarity": metrics["mean_target_similarity"],
                        "evaluated_sample_count": metrics["evaluated_sample_count"],
                    }
                )
                if rec_name == "adaface":
                    results.insert(0, "det_thresh", float(args.det_thresh))
                    results.insert(0, "verification_threshold", float(threshold["threshold"]))
                    results.insert(0, "recognizer", rec_name)
                    results.insert(0, "policy", policy)
                    results.insert(0, "forget_id", int(gid))
                    results.insert(0, "experiment_name", f"expanded_{gid}")
                    results.insert(0, "run_key", f"expanded_{gid}/{policy}")
                    adaface_prediction_frames.append(results)
                print(
                    f"  {gid}/{policy}/{rec_name}: FA={metrics['FA']:.2f} RA={metrics['RA']:.2f} "
                    f"selection_sim={selection_sim:.4f} residual_sim={metrics['mean_target_similarity']:.4f} "
                    f"n={metrics['evaluated_sample_count']}"
                )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_run_path = args.out_dir / "adaface_vs_arcface_per_run.csv"
    per_run = pd.DataFrame(rows)
    if per_run_path.exists() and set(args.recognizers) != {"arcface", "adaface"}:
        existing = pd.read_csv(per_run_path)
        existing = existing.loc[~existing["recognizer"].isin(args.recognizers)]
        per_run = pd.concat([existing, per_run], ignore_index=True)
    per_run["policy"] = pd.Categorical(per_run["policy"], categories=POLICIES, ordered=True)
    per_run["recognizer"] = pd.Categorical(
        per_run["recognizer"], categories=["arcface", "adaface"], ordered=True
    )
    per_run = per_run.sort_values(["group", "policy", "recognizer"]).reset_index(drop=True)
    per_run["policy"] = per_run["policy"].astype(str)
    per_run["recognizer"] = per_run["recognizer"].astype(str)
    if per_run.duplicated(["group", "policy", "recognizer"]).any():
        raise ValueError("Duplicate group/policy/recognizer rows after partial rescore")
    per_run.to_csv(per_run_path, index=False)

    if adaface_prediction_frames:
        adaface_predictions = pd.concat(adaface_prediction_frames, ignore_index=True)
        if adaface_predictions.duplicated(
            ["run_key", "role", "intended_identity", "reference_image", "seed"]
        ).any():
            raise ValueError("Duplicate AdaFace per-image prediction keys")
        adaface_predictions.to_csv(args.out_dir / "adaface_per_image.csv", index=False)

    summary_rows = []
    for (policy, rec), grp in per_run.groupby(["policy", "recognizer"]):
        summary_rows.append(
            {
                "policy": policy,
                "recognizer": rec,
                "num_groups": int(len(grp)),
                "FA0_groups": int((grp["FA"] == 0.0).sum()),
                "mean_FA": float(grp["FA"].mean()),
                "mean_RA": float(grp["RA"].mean()),
                "mean_ERB": float(grp["ERB"].mean()),
                "mean_IdentityLeak8": float(grp["IdentityLeak@8"].mean()),
                "mean_target_similarity": float(grp["mean_target_similarity"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["recognizer", "policy"])
    summary.to_csv(args.out_dir / "adaface_vs_arcface_summary.csv", index=False)

    try:
        from scipy import stats as _stats
    except Exception:  # pragma: no cover
        _stats = None

    def _corr(x, y) -> dict:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if len(x) < 3 or x.std() == 0 or y.std() == 0:
            return {"n": int(len(x)), "pearson_r": None, "pearson_p": None, "spearman_rho": None, "spearman_p": None}
        if _stats is not None:
            pe = _stats.pearsonr(x, y)
            sp = _stats.spearmanr(x, y)
            return {
                "n": int(len(x)),
                "pearson_r": float(pe.statistic), "pearson_p": float(pe.pvalue),
                "spearman_rho": float(sp.statistic), "spearman_p": float(sp.pvalue),
            }
        return {"n": int(len(x)), "pearson_r": float(np.corrcoef(x, y)[0, 1]), "pearson_p": None,
                "spearman_rho": None, "spearman_p": None}

    selection_dose = {}
    residual_dose = {}
    for rec in ["arcface", "adaface"]:
        sub = per_run[per_run["recognizer"] == rec]
        selection_dose[rec] = _corr(sub["selection_target_similarity"].to_numpy(), sub["FA"].to_numpy())
        residual_dose[rec] = _corr(sub["mean_target_similarity"].to_numpy(), sub["FA"].to_numpy())
    selection_dose["note"] = (
        "x = target_similarity_to_forget from adapter training; y = FA under each recognizer."
    )
    residual_dose["note"] = (
        "x = mean post-generation similarity to the forget centroid; y = FA under each recognizer."
    )

    arc = per_run[per_run["recognizer"] == "arcface"].set_index(["group", "policy"])["FA"]
    ada = per_run[per_run["recognizer"] == "adaface"].set_index(["group", "policy"])["FA"]
    paired = arc.to_frame("arcface_FA").join(ada.to_frame("adaface_FA"), how="inner").dropna()
    agreement = {
        "n_pairs": int(len(paired)),
        "fa_correlation": _corr(paired["arcface_FA"].to_numpy(), paired["adaface_FA"].to_numpy()),
        "mean_abs_fa_diff": float((paired["arcface_FA"] - paired["adaface_FA"]).abs().mean()) if len(paired) else None,
        "both_fa0_pairs": int(((paired["arcface_FA"] == 0) & (paired["adaface_FA"] == 0)).sum()),
        "either_fa0_pairs": int(((paired["arcface_FA"] == 0) | (paired["adaface_FA"] == 0)).sum()),
    }

    out = {
        "matched_groups": matched,
        "num_groups": len(matched),
        "selection_similarity_dose_response": selection_dose,
        "residual_similarity_dose_response": residual_dose,
        "cross_recognizer_agreement": agreement,
    }
    with (args.out_dir / "adaface_vs_arcface_dose_response.json").open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2)

    print(f"\n=== Matched groups: {len(matched)} ===")
    print("\n=== Per-policy summary (mean over matched groups) ===")
    print(summary.to_string(index=False))
    print("\n=== Pooled dose-response: selection-time target similarity vs FA (paper-consistent x-axis) ===")
    print(json.dumps(selection_dose, indent=2))
    print("\n=== Pooled dose-response: post-generation residual similarity vs FA (diagnostic) ===")
    print(json.dumps(residual_dose, indent=2))
    print("\n=== Cross-recognizer FA agreement ===")
    print(json.dumps(agreement, indent=2))
    print(f"\nWrote outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
