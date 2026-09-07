from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARCFACE_MODEL_ROOT = ROOT / "models" / "insightface"

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder, extract_real_image_embeddings, save_embedding_artifacts
from genmu_face_unlearning.benchmark import (
    build_benchmark,
    build_global_reference_maps,
    build_identity_centroids,
    compute_identity_neighbors,
    dump_benchmark,
    load_public_face_groups,
)
from genmu_face_unlearning.celeba import build_metadata
from genmu_face_unlearning.threshold import calibrate_arcface_threshold, save_threshold_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CelebA identity metadata, ArcFace embeddings, benchmark splits, and threshold calibration.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arcface-model-root", type=Path, default=DEFAULT_ARCFACE_MODEL_ROOT)
    parser.add_argument("--public-splits", type=Path, default=ROOT / "Data" / "validation-splits.json")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--min-images-per-identity", type=int, default=15)
    parser.add_argument("--expanded-forget-count", type=int, default=30)
    parser.add_argument("--fit-reference-count", type=int, default=3)
    parser.add_argument("--eval-reference-count", type=int, default=3)
    parser.add_argument("--partitions", type=int, nargs="*", default=None, help="Optional CelebA partitions to keep, e.g. 0 1 2")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap for a fast smoke test run.")
    args = parser.parse_args()

    start_time = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading CelebA metadata...")
    metadata = build_metadata(args.data_dir)
    if args.partitions is not None and len(args.partitions) > 0:
        metadata = metadata.loc[metadata["partition"].isin(args.partitions)].copy().reset_index(drop=True)
    print(f"Loaded {len(metadata):,} images across {metadata['identity'].nunique():,} identities")

    print("[2/6] Filtering eligible identities...")
    eligible = metadata.groupby("identity").filter(lambda frame: len(frame) >= args.min_images_per_identity).copy()
    eligible = eligible.sort_values(["identity", "image_name"]).reset_index(drop=True)
    if args.max_images is not None:
        eligible = eligible.head(args.max_images).copy()
    print(
        f"Eligible pool: {len(eligible):,} images from "
        f"{eligible['identity'].nunique():,} identities "
        f"(min_images_per_identity={args.min_images_per_identity})"
    )

    print("[3/6] Initializing ArcFace / InsightFace models...")
    embedder = ArcFaceEmbedder(model_root=args.arcface_model_root, device=args.device)
    print("[4/6] Extracting real-image ArcFace embeddings...")
    kept_metadata, embeddings = extract_real_image_embeddings(eligible, embedder)
    save_embedding_artifacts(kept_metadata, embeddings, args.out_dir, prefix="real")
    print(f"Extracted {len(kept_metadata):,} embeddings with dimension {embeddings.shape[1]}")

    print("[5/6] Splitting references, then building fit-only centroid neighbors and benchmark groups...")
    public_face_groups = load_public_face_groups(args.public_splits)
    full_identity_counts = (
        kept_metadata.groupby("identity", as_index=False)
        .size()
        .rename(columns={"size": "num_images"})
    )
    eligible_ids = full_identity_counts.loc[
        full_identity_counts["num_images"] >= args.min_images_per_identity,
        "identity",
    ].astype(int).tolist()
    _, global_eval_reference_images = build_global_reference_maps(
        metadata=kept_metadata,
        identities=eligible_ids,
        fit_reference_count=args.fit_reference_count,
        eval_reference_count=args.eval_reference_count,
    )
    eval_reference_paths = {
        str(path)
        for refs in global_eval_reference_images.values()
        for path in refs
    }
    train_pool_mask = ~kept_metadata["image_path"].isin(eval_reference_paths)
    train_pool_metadata = kept_metadata.loc[train_pool_mask].reset_index(drop=True)
    train_pool_embeddings = embeddings[train_pool_mask.to_numpy()]
    save_embedding_artifacts(train_pool_metadata, train_pool_embeddings, args.out_dir, prefix="real_trainpool")

    centroids = build_identity_centroids(train_pool_metadata, train_pool_embeddings)
    centroids = centroids.rename(columns={"num_images": "centroid_num_images"})
    centroids = centroids.merge(full_identity_counts, on="identity", how="left", validate="one_to_one")
    centroids = centroids[["identity", "num_images", "centroid_num_images", "centroid", "partition"]]
    public_ids = {int(group["forget_id"]) for group in public_face_groups}
    public_ids.update(
        int(identity) for group in public_face_groups for identity in group["retain_ids"]
    )
    selection_centroids = centroids.loc[~centroids["identity"].isin(public_ids)].copy()
    neighbors = compute_identity_neighbors(selection_centroids, top_k=100)
    neighbors.to_csv(args.out_dir / "identity_neighbors.csv", index=False)

    benchmark = build_benchmark(
        metadata=kept_metadata,
        centroids=centroids,
        neighbors=neighbors,
        min_images_per_identity=args.min_images_per_identity,
        public_face_groups=public_face_groups,
        expanded_forget_count=args.expanded_forget_count,
        fit_reference_count=args.fit_reference_count,
        eval_reference_count=args.eval_reference_count,
    )
    benchmark["centroid_construction_policy"] = {
        "split_before_centroid_construction": True,
        "embedding_pool": "all extracted eligible images except every identity's evaluation references",
        "evaluation_references_used_for_centroids": False,
        "evaluation_references_used_for_neighbors": False,
        "evaluation_references_used_for_cohort_selection": False,
    }
    dump_benchmark(benchmark, args.out_dir / "benchmark.json")

    print("[6/6] Calibrating ArcFace verification threshold...")
    calibration = calibrate_arcface_threshold(train_pool_metadata, train_pool_embeddings)
    save_threshold_calibration(calibration, args.out_dir / "threshold_arcface.json")

    elapsed = time.time() - start_time
    print(f"Wrote embeddings, neighbors, benchmark, and threshold calibration to {args.out_dir}")
    print(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    main()
