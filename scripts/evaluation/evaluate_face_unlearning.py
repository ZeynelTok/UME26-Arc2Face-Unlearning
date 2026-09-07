from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARCFACE_MODEL_ROOT = ROOT / "models" / "insightface"

from genmu_face_unlearning.config import load_experiment_config
from genmu_face_unlearning.generation_plan import (
    expected_generation_keys,
    manifest_generation_keys,
)
from genmu_face_unlearning.benchmark import load_benchmark, lookup_group
from genmu_face_unlearning.evaluation import evaluate_run, write_evaluation_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate matched before/after face-unlearning runs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--threshold", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arcface-model-root", type=Path, default=DEFAULT_ARCFACE_MODEL_ROOT)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--use-fallback", action="store_true")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    benchmark = load_benchmark(args.benchmark)
    group = lookup_group(benchmark, config["forget_id"])
    expected_keys = expected_generation_keys(config, group)
    manifest_path = args.after_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Generation manifest is missing: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    observed_keys = manifest_generation_keys(manifest)
    if observed_keys != expected_keys:
        raise ValueError("Generation manifest does not match the exact expected key plan")
    expected_paths = {str(row["relative_image_path"]) for row in expected_keys}
    observed_paths = {
        path.relative_to(args.after_dir).as_posix() for path in args.after_dir.glob("**/*.png")
    }
    if observed_paths != expected_paths:
        raise ValueError("Generated PNG files do not match the exact expected key plan")
    results, metrics = evaluate_run(
        config=config,
        benchmark_path=args.benchmark,
        threshold_path=args.threshold,
        after_dir=args.after_dir,
        arcface_model_root=args.arcface_model_root,
        device=args.device,
        det_thresh=args.det_thresh,
        use_fallback=args.use_fallback,
    )
    metrics["expected_generation_key_count"] = len(expected_keys)
    write_evaluation_outputs(results, metrics, args.out_dir)
    print(f"Wrote evaluation outputs to {args.out_dir}")

if __name__ == "__main__":
    main()
