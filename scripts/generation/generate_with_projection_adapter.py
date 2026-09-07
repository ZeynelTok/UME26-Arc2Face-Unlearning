from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARCFACE_MODEL_ROOT = ROOT / "models" / "insightface"

from genmu_face_unlearning.arc2face_runtime import Arc2FaceRuntime
from genmu_face_unlearning.generation_plan import (
    expected_generation_keys,
    manifest_generation_keys,
)
from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.benchmark import group_reference_list, load_benchmark, lookup_group
from genmu_face_unlearning.config import load_experiment_config
from genmu_face_unlearning.embedding_adapter import apply_projection_adapter, load_projection_adapter
from genmu_face_unlearning.generation import generate_identity_plan_images, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate matched Arc2Face outputs with a learned embedding projection adapter.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arcface-model-root", type=Path, default=DEFAULT_ARCFACE_MODEL_ROOT)
    parser.add_argument(
        "--reference-det-thresh",
        type=float,
        default=0.5,
        help="InsightFace detection threshold for clean conditioning-reference images.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    benchmark = load_benchmark(args.benchmark)
    group = lookup_group(benchmark, config["forget_id"])
    adapter = load_projection_adapter(args.adapter)
    expected_keys = expected_generation_keys(config, group)

    embedder = ArcFaceEmbedder(
        model_root=args.arcface_model_root,
        device=config["generator"]["device"],
        det_thresh=args.reference_det_thresh,
    )
    runtime = Arc2FaceRuntime(config["generator"])

    manifest = {
        "variant": args.out_dir.name,
        "adapter_path": str(args.adapter),
        "forget_id": config["forget_id"],
        "retain_ids": config["retain_ids"],
        "hard_retain_ids": config.get("hard_retain_ids", config["retain_ids"]),
        "random_retain_ids": config.get("random_retain_ids", []),
        "target_identity": adapter["target_identity"],
        "rank": adapter["rank"],
        "reference_det_thresh": float(args.reference_det_thresh),
        "generation_reference_det_thresh": float(args.reference_det_thresh),
        "eval_seeds": config["eval_seeds"],
        "generator": config["generator"],
        "generation": config["generation"],
        "expected_generation_key_count": len(expected_keys),
        "images": [],
    }

    def transform(identity, role, reference_path, face_embedding):
        projected_embedding, cosine_to_forget, residual_norm = apply_projection_adapter(face_embedding, adapter)
        return projected_embedding, {
            "cosine_to_forget_centroid": float(cosine_to_forget),
            "projection_residual_norm": float(residual_norm),
        }

    manifest["images"] = generate_identity_plan_images(
        runtime=runtime,
        embedder=embedder,
        out_dir=args.out_dir,
        config=config,
        reference_lookup=lambda identity: group_reference_list(group, identity, split="fit"),
        transform=transform,
    )
    observed_keys = manifest_generation_keys(manifest)
    if observed_keys != expected_keys:
        raise RuntimeError("Generated image records do not match the exact expected key plan")

    write_manifest(args.out_dir / "manifest.json", manifest)
    print(f"Wrote projection-adapter generations to {args.out_dir}")


if __name__ == "__main__":
    main()
