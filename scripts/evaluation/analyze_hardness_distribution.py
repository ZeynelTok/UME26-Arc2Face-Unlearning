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

from genmu_face_unlearning.benchmark import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the full hardness-score distribution (mean cosine distance to top-3 centroid "
        "neighbors) across the entire eligible non-public identity pool, to check whether the top-10/20/30 "
        "expanded-benchmark cutoffs sit at a natural, data-driven boundary rather than an arbitrary round number."
    )
    parser.add_argument("--prep-dir", type=Path, default=ROOT / "outputs" / "prep")
    parser.add_argument("--base-benchmark", type=Path, default=ROOT / "outputs" / "prep" / "benchmark.json")
    parser.add_argument("--min-images-per-identity", type=int, default=15)
    parser.add_argument("--hard-retain-count", type=int, default=3)
    parser.add_argument("--top-k-neighbors", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sensitivity" / "hardness_distribution")
    args = parser.parse_args()

    base_benchmark = load_benchmark(args.base_benchmark)
    public_ids = {int(group["forget_id"]) for group in base_benchmark.get("public_groups", [])}
    public_ids.update(int(x) for group in base_benchmark.get("public_groups", []) for x in group.get("retain_ids", []))

    # Reuse the split-consistent graph so evaluation references stay out of the
    # identity centroids.
    neighbors = pd.read_csv(args.prep_dir / "identity_neighbors.csv")
    if int(neighbors["rank"].max()) < args.top_k_neighbors:
        raise ValueError(
            f"Saved neighbor graph has ranks only through {int(neighbors['rank'].max())}, "
            f"but --top-k-neighbors={args.top_k_neighbors} was requested"
        )
    if neighbors["target_identity"].isin(public_ids).any() or neighbors[
        "retain_identity"
    ].isin(public_ids).any():
        raise ValueError(
            "Saved neighbor graph contains public identities. Rebuild outputs/prep/identity_neighbors.csv "
            "with scripts/data_prep/prepare_identities.py before running this analysis."
        )
    candidate_neighbors = neighbors.loc[
        (neighbors["target_num_images"] >= args.min_images_per_identity)
        & (neighbors["retain_num_images"] >= args.min_images_per_identity)
    ].copy()

    top_neighbors = candidate_neighbors.loc[
        candidate_neighbors["rank"] <= args.hard_retain_count
    ]
    top_counts = top_neighbors.groupby("target_identity").size()
    if len(top_counts) != candidate_neighbors["target_identity"].nunique() or not (
        top_counts == args.hard_retain_count
    ).all():
        raise ValueError(
            f"Every eligible non-public identity must have exactly {args.hard_retain_count} "
            "reranked non-public neighbors"
        )

    hardness = (
        top_neighbors
        .groupby("target_identity", as_index=False)["cosine_distance"]
        .mean()
        .rename(columns={"cosine_distance": "mean_topk_distance"})
        .sort_values(["mean_topk_distance", "target_identity"], ascending=[True, True])
        .reset_index(drop=True)
    )
    hardness["rank"] = np.arange(1, len(hardness) + 1)

    n_total = len(hardness)
    scores = hardness["mean_topk_distance"].to_numpy()

    pool_stats = {
        "n_eligible_non_public_identities": int(n_total),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "percentiles": {p: float(np.percentile(scores, p)) for p in [1, 2, 5, 10, 25, 50, 75, 90]},
    }

    cutoff_report = {}
    for k in [10, 20, 30, 40, 50, 75, 100]:
        if k > n_total:
            continue
        cutoff_value = float(hardness["mean_topk_distance"].iloc[k - 1])
        z_score = (cutoff_value - pool_stats["mean"]) / pool_stats["std"]
        percentile_rank = 100.0 * k / n_total
        cutoff_report[k] = {
            "cutoff_distance": cutoff_value,
            "z_score_vs_pool": z_score,
            "percentile_of_pool": percentile_rank,
        }

    gaps = np.diff(scores)
    gap_records = [
        {"between_rank": int(i + 1), "and_rank": int(i + 2), "gap": float(gaps[i])}
        for i in range(len(gaps))
    ]
    top_gap_region = sorted(gap_records[:120], key=lambda r: -r["gap"])[:15]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hardness.to_csv(args.out_dir / "full_hardness_ranking.csv", index=False)
    with (args.out_dir / "hardness_distribution_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "pool_stats": pool_stats,
                "cutoff_report": cutoff_report,
                "largest_gaps_within_first_120_ranks": top_gap_region,
                "neighbor_graph_validation": {
                    "public_identity_count": len(public_ids),
                    "public_rows_present": 0,
                    "neighbors_per_identity": int(args.hard_retain_count),
                    "validated_identity_count": int(n_total),
                },
            },
            handle,
            indent=2,
        )

    print(json.dumps({"pool_stats": pool_stats, "cutoff_report": cutoff_report}, indent=2))
    print("\nLargest rank-to-rank gaps within the first 120 ranks (candidate natural elbows):")
    for record in top_gap_region[:10]:
        print(f"  rank {record['between_rank']} -> {record['and_rank']}: gap={record['gap']:.6f}")
    print(f"\nWrote {args.out_dir / 'hardness_distribution_summary.json'}")
    print(f"Wrote {args.out_dir / 'full_hardness_ranking.csv'}")


if __name__ == "__main__":
    main()
