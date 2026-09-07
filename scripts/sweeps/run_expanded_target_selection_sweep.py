from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARCFACE_MODEL_ROOT = ROOT / "models" / "insightface"
DEFAULT_BENCHMARK = ROOT / "outputs" / "prep" / "benchmark.json"
DEFAULT_THRESHOLD = ROOT / "outputs" / "prep" / "threshold_arcface.json"

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.generation_plan import (
    expected_generation_keys,
    manifest_generation_keys,
)
from genmu_face_unlearning.benchmark import (
    group_reference_list,
    group_reference_map,
    load_benchmark,
    lookup_group,
)
from genmu_face_unlearning.config import load_experiment_config
from genmu_face_unlearning.embedding_adapter import (
    centroid,
    load_projection_adapter,
    select_identity_embedding,
)


def _reference_embeddings(embedder: ArcFaceEmbedder, reference_paths: list[str]) -> list[np.ndarray]:
    embeddings = []
    for path in reference_paths:
        detection = embedder.detect_and_embed_image(path)
        if detection is None:
            raise RuntimeError(f"Unable to detect face in reference image: {path}")
        embeddings.append(detection.embedding)
    return embeddings


def _choose_target(
    *,
    config: dict,
    benchmark: dict,
    embedder: ArcFaceEmbedder,
    metadata_path: Path,
    embeddings_path: Path,
    policy: str,
    random_seed: int,
) -> dict:
    group = lookup_group(benchmark, config["forget_id"])
    forget_paths = group_reference_list(group, config["forget_id"], split="fit")
    forget_centroid = centroid(_reference_embeddings(embedder, forget_paths))

    candidates = []
    for identity in config["hard_retain_ids"]:
        target_identity, _, similarity = select_identity_embedding(
            forget_embedding=forget_centroid,
            forget_id=config["forget_id"],
            benchmark=benchmark,
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            mode="nearest",
            candidate_ids=[int(identity)],
        )
        candidates.append(
            {
                "identity": int(target_identity),
                "similarity": float(similarity),
            }
        )

    if policy == "nearest-hard":
        chosen = max(candidates, key=lambda item: item["similarity"])
    elif policy == "least-sim-hard":
        chosen = min(candidates, key=lambda item: item["similarity"])
    elif policy == "median-hard":
        ordered = sorted(candidates, key=lambda item: item["similarity"])
        chosen = ordered[len(ordered) // 2]
    else:
        rng = np.random.default_rng(random_seed + int(config["forget_id"]))
        chosen = candidates[int(rng.integers(len(candidates)))]

    return {
        "forget_id": int(config["forget_id"]),
        "policy": policy,
        "target_identity": int(chosen["identity"]),
        "target_similarity_to_forget": float(chosen["similarity"]),
        "candidates": candidates,
    }


def _run(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=str(ROOT))


def _script_args(script_relative_path: list[str], *args: str) -> list[str]:
    return [sys.executable, str(ROOT.joinpath(*script_relative_path)), *args]


def _has_complete_candidate_evidence(row: dict) -> bool:
    candidates = row.get("candidates", row.get("candidate_similarities", []))
    if not isinstance(candidates, list) or len(candidates) != 3:
        return False
    try:
        identities = [int(candidate["identity"]) for candidate in candidates]
        similarities = [float(candidate["similarity"]) for candidate in candidates]
        target_identity = int(row["target_identity"])
        target_similarity = float(row["target_similarity_to_forget"])
    except (KeyError, TypeError, ValueError):
        return False
    if len(set(identities)) != 3 or not all(np.isfinite(similarities)):
        return False
    chosen = [
        similarity
        for identity, similarity in zip(identities, similarities)
        if identity == target_identity
    ]
    return len(chosen) == 1 and bool(np.isclose(chosen[0], target_similarity, rtol=0.0, atol=1e-7))


def _load_existing_selections(summary_path: Path, policy: str) -> dict[str, dict]:
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("policy") != policy:
        return {}
    return {
        str(row["experiment_name"]): row
        for row in summary.get("rows", [])
        if _has_complete_candidate_evidence(row)
    }


def _selection_from_adapter_summary(adapter_dir: Path, config: dict, policy: str) -> dict | None:
    summary_path = adapter_dir / "projection_adapter_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    target_identity = summary.get("target_identity")
    target_similarity = summary.get("target_similarity_to_forget")
    if target_identity is None or target_similarity is None:
        return None
    selection = {
        "forget_id": int(config["forget_id"]),
        "policy": policy,
        "target_identity": int(target_identity),
        "target_similarity_to_forget": float(target_similarity),
        "candidates": summary.get("candidate_similarities", []),
    }
    return selection if _has_complete_candidate_evidence(selection) else None


def _run_is_compatible(
    *,
    adapter_dir: Path,
    generation_dir: Path,
    eval_dir: Path,
    config: dict,
    benchmark_path: Path,
    threshold_path: Path,
    group: dict,
    expected_target_identity: int,
    reference_det_thresh: float,
    evaluation_det_thresh: float,
) -> bool:
    """Return whether saved outputs match the requested run."""
    adapter_path = adapter_dir / "projection_adapter.npz"
    summary_path = adapter_dir / "projection_adapter_summary.json"
    manifest_path = generation_dir / "manifest.json"
    metrics_path = eval_dir / "metrics.json"
    per_sample_path = eval_dir / "per_sample_results.csv"
    gallery_reference_path = benchmark_path.parent / "gallery_reference_detections.csv"
    required = [
        adapter_path,
        summary_path,
        manifest_path,
        metrics_path,
        per_sample_path,
        gallery_reference_path,
    ]
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        return False
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        with threshold_path.open("r", encoding="utf-8") as handle:
            threshold = json.load(handle)
        adapter = load_projection_adapter(adapter_path)
    except (OSError, ValueError, TypeError, KeyError):
        return False

    expected_forget = int(config["forget_id"])
    if summary.get("checkpoint_selection_timing") != "pre_update_state_scored_and_saved_before_optimizer_step":
        return False
    if int(summary.get("forget_id", -1)) != expected_forget:
        return False
    if int(summary.get("target_identity", -1)) != int(expected_target_identity):
        return False
    if int(summary.get("rank", -1)) != 8:
        return False
    if int(adapter.get("target_identity", -1)) != int(expected_target_identity):
        return False
    if int(adapter.get("rank", -1)) != 8:
        return False
    if adapter["down"].shape != (8, 512) or adapter["up"].shape != (512, 8):
        return False
    if int(manifest.get("forget_id", -1)) != expected_forget:
        return False
    if int(manifest.get("target_identity", -1)) != int(expected_target_identity):
        return False
    if int(manifest.get("rank", -1)) != 8:
        return False
    recorded_adapter_path = Path(str(manifest.get("adapter_path", "")))
    if not recorded_adapter_path.is_absolute():
        recorded_adapter_path = ROOT / recorded_adapter_path
    if recorded_adapter_path.resolve() != adapter_path.resolve():
        return False
    if [int(value) for value in manifest.get("eval_seeds", [])] != config["eval_seeds"]:
        return False
    for field in ("retain_ids", "hard_retain_ids", "random_retain_ids"):
        if field in manifest and [int(value) for value in manifest[field]] != config[field]:
            return False
    if "generator" in manifest and manifest["generator"] != config["generator"]:
        return False
    if "generation" in manifest and manifest["generation"] != config["generation"]:
        return False
    expected_keys = expected_generation_keys(config, group)
    if len(expected_keys) != 168:
        return False
    observed_keys = manifest_generation_keys(manifest)
    if observed_keys != expected_keys or len({tuple(row.values()) for row in observed_keys}) != 168:
        return False

    expected_relative_paths = {row["relative_image_path"] for row in expected_keys}
    actual_relative_paths = {
        path.relative_to(generation_dir).as_posix() for path in generation_dir.glob("**/*.png")
    }
    if actual_relative_paths != expected_relative_paths:
        return False
    for image, key in zip(manifest["images"], observed_keys, strict=True):
        recorded = Path(str(image["image_path"]))
        if not recorded.is_absolute():
            recorded = ROOT / recorded
        if recorded.resolve() != (generation_dir / str(key["relative_image_path"])).resolve():
            return False
    if float(summary.get("reference_det_thresh", -1.0)) != float(reference_det_thresh):
        return False
    if float(manifest.get("reference_det_thresh", -1.0)) != float(reference_det_thresh):
        return False
    if float(metrics.get("det_thresh", -1.0)) != float(evaluation_det_thresh):
        return False
    if float(metrics.get("threshold", -1.0)) != float(threshold.get("threshold", -2.0)):
        return False
    if int(metrics.get("attempted_sample_count", -1)) != 168:
        return False

    expected_short_keys = {
        (row["role"], int(row["identity"]), row["reference_stem"], int(row["seed"]))
        for row in expected_keys
    }
    try:
        with per_sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
            prediction_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    prediction_keys = {
        (
            str(row["role"]),
            int(row["intended_identity"]),
            Path(str(row["reference_image"])).stem,
            int(row["seed"]),
        )
        for row in prediction_rows
    }
    if len(prediction_keys) != len(prediction_rows) or not prediction_keys.issubset(
        expected_short_keys
    ):
        return False
    if len(prediction_rows) != int(metrics.get("evaluated_sample_count", -1)):
        return False

    try:
        with gallery_reference_path.open("r", encoding="utf-8-sig", newline="") as handle:
            gallery_rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return False
    expected_gallery = {
        (int(identity), Path(path).resolve())
        for identity, paths in group_reference_map(group, split="eval").items()
        for path in paths
    }
    observed_gallery = {
        (int(row["identity"]), Path(row["reference_image"]).resolve()) for row in gallery_rows
    }
    if (
        len(gallery_rows) != 480
        or len(observed_gallery) != 480
        or not expected_gallery.issubset(observed_gallery)
        or any(str(row.get("detector_success", "")).lower() != "true" for row in gallery_rows)
    ):
        return False
    return True


def _projection_pipeline_commands(
    *,
    config_path: Path,
    benchmark_path: Path,
    threshold_path: Path,
    metadata_path: Path,
    embeddings_path: Path,
    arcface_model_root: Path,
    adapter_dir: Path,
    generation_dir: Path,
    eval_dir: Path,
    policy_slug: str,
    target_identity: int,
    evaluation_device: str,
    reference_det_thresh: float,
    evaluation_det_thresh: float,
) -> list[list[str]]:
    return [
        _script_args(
            ["scripts", "training", "train_embedding_projection_adapter.py"],
            "--config",
            str(config_path),
            "--benchmark",
            str(benchmark_path),
            "--metadata",
            str(metadata_path),
            "--embeddings",
            str(embeddings_path),
            "--out-dir",
            str(adapter_dir),
            "--target-identity",
            str(target_identity),
            "--arcface-model-root",
            str(arcface_model_root),
            "--reference-det-thresh",
            str(reference_det_thresh),
        ),
        _script_args(
            ["scripts", "generation", "generate_with_projection_adapter.py"],
            "--config",
            str(config_path),
            "--benchmark",
            str(benchmark_path),
            "--adapter",
            str(adapter_dir / "projection_adapter.npz"),
            "--out-dir",
            str(generation_dir),
            "--arcface-model-root",
            str(arcface_model_root),
            "--reference-det-thresh",
            str(reference_det_thresh),
        ),
        _script_args(
            ["scripts", "evaluation", "evaluate_face_unlearning.py"],
            "--config",
            str(config_path),
            "--benchmark",
            str(benchmark_path),
            "--threshold",
            str(threshold_path),
            "--after-dir",
            str(generation_dir),
            "--out-dir",
            str(eval_dir),
            "--arcface-model-root",
            str(arcface_model_root),
            "--device",
            evaluation_device,
            "--det-thresh",
            str(evaluation_det_thresh),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed target-selection policy sweep over expanded projection-adapter groups.")
    parser.add_argument("--configs-dir", type=Path, default=ROOT / "configs" / "expanded")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--threshold", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--metadata", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_metadata.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_embeddings.npy")
    parser.add_argument("--arcface-model-root", type=Path, default=DEFAULT_ARCFACE_MODEL_ROOT)
    parser.add_argument("--policy", choices=["nearest-hard", "least-sim-hard", "median-hard", "random-hard"], default="least-sim-hard")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--evaluation-device",
        "--device",
        dest="evaluation_device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device used by the held-out evaluation stage; training and generation use the config device.",
    )
    parser.add_argument(
        "--reference-det-thresh",
        type=float,
        default=0.5,
        help="Detection threshold for clean fitting/conditioning references.",
    )
    parser.add_argument(
        "--evaluation-det-thresh",
        "--det-thresh",
        dest="evaluation_det_thresh",
        type=float,
        default=0.1,
        help="Detection threshold for held-out evaluation galleries and generated outputs.",
    )
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--selection-summary",
        type=Path,
        default=None,
        help="Optional frozen selection-only summary to consume without recomputing targets.",
    )
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Resolve and record every policy target without training, generation, or evaluation.",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark)
    centroid_policy = benchmark.get("centroid_construction_policy", {})
    if not (
        centroid_policy.get("split_before_centroid_construction") is True
        and centroid_policy.get("evaluation_references_used_for_centroids") is False
        and centroid_policy.get("evaluation_references_used_for_neighbors") is False
        and centroid_policy.get("evaluation_references_used_for_cohort_selection") is False
    ):
        raise ValueError(
            "Benchmark is not marked as split-first. Rebuild outputs/prep with "
            "scripts/data_prep/prepare_identities.py before running the paper sweep."
        )
    config_paths = sorted(args.configs_dir.glob("expanded_*.json"))
    if not config_paths:
        raise FileNotFoundError(f"No expanded configs found in {args.configs_dir}")
    benchmark_forget_ids = {int(group["forget_id"]) for group in benchmark.get("expanded_groups", [])}
    config_forget_ids = {int(load_experiment_config(path)["forget_id"]) for path in config_paths}
    if config_forget_ids != benchmark_forget_ids or len(config_paths) != len(benchmark_forget_ids):
        raise ValueError(
            "Expanded configs do not exactly match the benchmark cohort. Re-run "
            "scripts/configs/build_group_configs.py against the current benchmark."
        )

    policy_slug = {
        "nearest-hard": "nearest_hard",
        "least-sim-hard": "least_sim_hard",
        "median-hard": "median_hard",
        "random-hard": "random_hard",
    }[args.policy]
    summary_path = args.out_root / f"expanded_target_selection_{policy_slug}_summary.json"
    if args.selection_summary is not None:
        existing_selections = _load_existing_selections(args.selection_summary, args.policy)
        if len(existing_selections) != len(config_paths):
            raise ValueError(
                f"Frozen selection summary has {len(existing_selections)} rows; expected {len(config_paths)}"
            )
    else:
        existing_selections = _load_existing_selections(summary_path, args.policy) if args.skip_if_exists else {}

    sample_config = load_experiment_config(config_paths[0])
    embedder: ArcFaceEmbedder | None = None

    def choose_target(config: dict) -> dict:
        nonlocal embedder
        if embedder is None:
            embedder = ArcFaceEmbedder(
                model_root=args.arcface_model_root,
                device=sample_config["generator"]["device"],
                det_thresh=args.reference_det_thresh,
            )
        return _choose_target(
            config=config,
            benchmark=benchmark,
            embedder=embedder,
            metadata_path=args.metadata,
            embeddings_path=args.embeddings,
            policy=args.policy,
            random_seed=int(args.random_seed),
        )

    sweep_rows = []
    sweep_started = perf_counter()
    progress = tqdm(config_paths, desc=f"expanded_target_sweep:{policy_slug}", unit="group", dynamic_ncols=True)
    for config_path in progress:
        config = load_experiment_config(config_path)
        group = lookup_group(benchmark, config["forget_id"])
        experiment_name = str(config["experiment_name"])
        group_out_root = args.out_root / experiment_name
        adapter_dir = group_out_root / f"u_projection_adapter_{policy_slug}"
        generation_dir = group_out_root / f"{policy_slug}_gen"
        eval_dir = group_out_root / f"eval_{policy_slug}"
        metrics_path = eval_dir / "metrics.json"

        selection = existing_selections.get(experiment_name)
        if args.selection_summary is not None and selection is None:
            raise KeyError(f"Frozen selection summary has no row for {experiment_name}")
        if selection is None and args.skip_if_exists and metrics_path.exists():
            candidate_selection = _selection_from_adapter_summary(adapter_dir, config, args.policy)
            if candidate_selection is not None and _run_is_compatible(
                adapter_dir=adapter_dir,
                generation_dir=generation_dir,
                eval_dir=eval_dir,
                config=config,
                benchmark_path=args.benchmark,
                threshold_path=args.threshold,
                group=group,
                expected_target_identity=int(candidate_selection["target_identity"]),
                reference_det_thresh=float(args.reference_det_thresh),
                evaluation_det_thresh=float(args.evaluation_det_thresh),
            ):
                selection = candidate_selection
        if selection is None:
            selection = choose_target(config)

        progress.set_postfix(
            forget=config["forget_id"],
            target=selection["target_identity"],
            sim=f"{selection['target_similarity_to_forget']:.4f}",
        )

        if args.selection_only:
            sweep_rows.append(
                {
                    "experiment_name": experiment_name,
                    "forget_id": int(config["forget_id"]),
                    "policy": args.policy,
                    "target_identity": selection["target_identity"],
                    "target_similarity_to_forget": selection["target_similarity_to_forget"],
                    "candidate_similarities": selection.get("candidates", []),
                }
            )
            continue

        reuse_existing = args.skip_if_exists and _run_is_compatible(
            adapter_dir=adapter_dir,
            generation_dir=generation_dir,
            eval_dir=eval_dir,
            config=config,
            benchmark_path=args.benchmark,
            threshold_path=args.threshold,
            group=group,
            expected_target_identity=int(selection["target_identity"]),
            reference_det_thresh=float(args.reference_det_thresh),
            evaluation_det_thresh=float(args.evaluation_det_thresh),
        )
        if not reuse_existing:
            commands = _projection_pipeline_commands(
                config_path=config_path,
                benchmark_path=args.benchmark,
                threshold_path=args.threshold,
                metadata_path=args.metadata,
                embeddings_path=args.embeddings,
                arcface_model_root=args.arcface_model_root,
                adapter_dir=adapter_dir,
                generation_dir=generation_dir,
                eval_dir=eval_dir,
                policy_slug=policy_slug,
                target_identity=int(selection["target_identity"]),
                evaluation_device=args.evaluation_device,
                reference_det_thresh=float(args.reference_det_thresh),
                evaluation_det_thresh=float(args.evaluation_det_thresh),
            )
            for command in commands:
                _run(command)

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        sweep_rows.append(
            {
                "experiment_name": experiment_name,
                "forget_id": int(config["forget_id"]),
                "policy": args.policy,
                "target_identity": selection["target_identity"],
                "target_similarity_to_forget": selection["target_similarity_to_forget"],
                "candidate_similarities": selection.get("candidates", selection.get("candidate_similarities", [])),
                "fa": metrics.get("FA"),
                "ra": metrics.get("RA"),
                "erb": metrics.get("ERB"),
                "identity_leak_at_8": metrics.get("IdentityLeak@8"),
                "evaluated": metrics.get("evaluated_sample_count"),
                "dropped": metrics.get("dropped_no_face_count"),
                "metrics_path": str(metrics_path),
            }
        )

    summary = {
        "policy": args.policy,
        "selection_only": bool(args.selection_only),
        "group_count": len(sweep_rows),
        "elapsed_seconds": perf_counter() - sweep_started,
        "rows": sweep_rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
