from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.benchmark import build_identity_centroids, iter_groups, load_benchmark
from genmu_face_unlearning.evaluation import _collect_images
from genmu_face_unlearning.threshold import load_threshold_calibration

EXPANDED_POLICIES = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]


def _build_group_lookup(benchmark_paths: list[Path]) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    for path in benchmark_paths:
        if not path.exists():
            continue
        benchmark = load_benchmark(path)
        for group in iter_groups(benchmark):
            forget_id = int(group["forget_id"])
            if forget_id not in lookup:
                lookup[forget_id] = group
    return lookup


def _discover_expanded_runs(out_root: Path) -> list[dict]:
    runs = []
    for group_dir in sorted(out_root.glob("expanded_*")):
        if not group_dir.is_dir():
            continue
        for policy in EXPANDED_POLICIES:
            gen_dir = group_dir / f"{policy}_gen"
            summary_path = group_dir / f"u_projection_adapter_{policy}" / "projection_adapter_summary.json"
            if gen_dir.exists() and summary_path.exists():
                runs.append({"gen_dir": gen_dir, "summary_path": summary_path, "policy": policy, "group_name": group_dir.name})
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether already-generated redirected images verify as a real identity "
        "outside the benchmark group (large-gallery bystander leakage), not just as the forget or target identity."
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--metadata", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_metadata.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_embeddings.npy")
    parser.add_argument("--threshold", type=Path, default=ROOT / "outputs" / "prep" / "threshold_arcface.json")
    parser.add_argument(
        "--benchmarks",
        type=Path,
        nargs="+",
        default=[ROOT / "outputs" / "prep" / "benchmark.json"],
    )
    parser.add_argument("--arcface-model-root", type=Path, default=ROOT / "models" / "insightface")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "large_gallery_leakage")
    args = parser.parse_args()

    print("Loading real-image pool and building large-gallery identity centroids...")
    metadata = pd.read_csv(args.metadata)
    embeddings = np.load(args.embeddings)
    centroids_frame = build_identity_centroids(metadata, embeddings)
    identities = centroids_frame["identity"].to_numpy()
    centroid_matrix = np.stack(centroids_frame["centroid"].to_list(), axis=0).astype(np.float32)
    print(f"Large gallery: {len(identities)} identities")

    threshold = load_threshold_calibration(args.threshold)["threshold"]
    group_lookup = _build_group_lookup(args.benchmarks)

    runs = _discover_expanded_runs(args.out_root)
    if not runs:
        raise FileNotFoundError("No generated runs discovered under the given out-root.")
    print(f"Discovered {len(runs)} runs to check.")

    embedder = ArcFaceEmbedder(model_root=args.arcface_model_root, device=args.device, det_thresh=0.1)

    all_rows = []
    for run in runs:
        with run["summary_path"].open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        forget_id = int(summary["forget_id"])
        target_identity = int(summary["target_identity"])
        group = group_lookup.get(forget_id, {})
        in_group_ids = {forget_id, target_identity}
        in_group_ids.update(int(x) for x in group.get("hard_retain_ids", []))
        in_group_ids.update(int(x) for x in group.get("random_retain_ids", []))
        in_group_ids.update(int(x) for x in group.get("retain_ids", []))

        images = _collect_images(run["gen_dir"])
        if images.empty:
            continue

        for row in tqdm(images.to_dict("records"), desc=f"{run['group_name']}:{run['policy']}", unit="img", dynamic_ncols=True):
            detection = embedder.detect_and_embed_image(row["image_path"])
            if detection is None:
                continue
            similarities = centroid_matrix @ detection.embedding
            best_index = int(np.argmax(similarities))
            best_identity = int(identities[best_index])
            best_similarity = float(similarities[best_index])
            verifies = best_similarity >= threshold
            bystander = bool(verifies and best_identity not in in_group_ids)
            all_rows.append(
                {
                    "group_name": run["group_name"],
                    "policy": run["policy"],
                    "role": row["role"],
                    "intended_identity": row["intended_identity"],
                    "forget_id": forget_id,
                    "target_identity": target_identity,
                    "image_path": row["image_path"],
                    "large_gallery_predicted_identity": best_identity,
                    "large_gallery_best_similarity": best_similarity,
                    "verifies_large_gallery": verifies,
                    "bystander_leak": bystander,
                }
            )

    results = pd.DataFrame(all_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_dir / "large_gallery_per_image.csv", index=False)

    summary_rows = []
    for (policy, role), group in results.groupby(["policy", "role"]):
        summary_rows.append(
            {
                "policy": policy,
                "role": role,
                "num_images": int(len(group)),
                "bystander_leak_count": int(group["bystander_leak"].sum()),
                "bystander_leak_rate": 100.0 * float(group["bystander_leak"].mean()),
            }
        )
    summary_frame = pd.DataFrame(summary_rows).sort_values(["policy", "role"])
    summary_frame.to_csv(args.out_dir / "large_gallery_summary.csv", index=False)

    overall = {
        "total_images_checked": int(len(results)),
        "total_bystander_leaks": int(results["bystander_leak"].sum()),
        "overall_bystander_leak_rate": 100.0 * float(results["bystander_leak"].mean()) if len(results) else None,
        "forget_role_bystander_leak_rate": 100.0 * float(results.loc[results["role"] == "forget", "bystander_leak"].mean())
        if (results["role"] == "forget").any()
        else None,
        "retain_role_bystander_leak_rate": 100.0 * float(results.loc[results["role"] == "retain", "bystander_leak"].mean())
        if (results["role"] == "retain").any()
        else None,
        "large_gallery_identity_count": int(len(identities)),
    }
    with (args.out_dir / "large_gallery_overall_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(overall, handle, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"Wrote {args.out_dir / 'large_gallery_overall_summary.json'}")
    print(f"Wrote {args.out_dir / 'large_gallery_summary.csv'}")
    print(f"Wrote {args.out_dir / 'large_gallery_per_image.csv'}")


if __name__ == "__main__":
    main()
