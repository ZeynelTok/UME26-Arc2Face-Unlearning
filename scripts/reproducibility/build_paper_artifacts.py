"""Build the paper artifact bundle from completed experiment outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reproducibility.recompute_paper_tables import latex_constants, recompute
from genmu_face_unlearning.generation_plan import (
    expected_generation_keys,
    manifest_generation_keys,
)
from genmu_face_unlearning.benchmark import lookup_group
from genmu_face_unlearning.config import load_experiment_config


POLICIES = ("nearest_hard", "random_hard", "median_hard", "least_sim_hard")
POLICY_LABELS = {policy: policy.replace("_", "-") for policy in POLICIES}
BUNDLE_FILENAMES = {
    "README.md",
    "adaface_predictions.csv.gz",
    "adapter_summaries.jsonl",
    "adapters.npz",
    "generation_attempts.csv.gz",
    "generation_runs.jsonl",
    "gallery_reference_detections.csv",
    "hardness_top3_neighbors.csv.gz",
    "large_gallery_predictions.csv.gz",
    "paper_table_values.tex",
    "per_sample_predictions.csv.gz",
    "protocol.json",
    "quality_scores.csv.gz",
    "recognizer_runs.csv",
    "reported_summaries.json",
    "run_metrics.csv",
    "sensitivity_policy_metrics.csv",
    "target_candidates.csv",
    "target_selections.csv",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def portable(value: Any, root: Path) -> Any:
    """Recursively remove machine-specific workspace prefixes and separators."""
    if isinstance(value, dict):
        return {key: portable(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item, root) for item in value]
    if not isinstance(value, str):
        return value
    root_text = str(root.resolve())
    text = value
    if text.lower().startswith(root_text.lower() + "\\"):
        text = text[len(root_text) + 1 :]
    elif text.lower().startswith(root_text.lower() + "/"):
        text = text[len(root_text) + 1 :]
    if text.startswith(("outputs\\", "Data\\", "models\\", "configs\\")):
        return text.replace("\\", "/")
    return text


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    # pandas forwards mtime=0 to gzip, making repeated builds byte-identical.
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
        lineterminator="\n",
    )


def normalize_frame_paths(frame: pd.DataFrame, root: Path) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        if "path" in column or column in {"image_path", "reference_image", "after_dir"}:
            result[column] = result[column].map(lambda item: portable(item, root))
    return result


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    return path


def collect_small_summaries(root: Path) -> dict[str, Any]:
    patterns = (
        "outputs/expanded_summary/*.json",
        "outputs/quality_official_v1/comparison.json",
        "outputs/quality_official_v1/summary.json",
        "outputs/adaface_verification/*.json",
        "outputs/large_gallery_leakage/*.json",
        "outputs/sensitivity/comparison/*.json",
    )
    summaries: dict[str, Any] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            summaries[path.relative_to(root).as_posix()] = portable(read_json(path), root)

    csv_paths = (
        root / "outputs/adaface_verification/adaface_vs_arcface_summary.csv",
        root / "outputs/large_gallery_leakage/large_gallery_summary.csv",
        root / "outputs/sensitivity/comparison/policy_metrics.csv",
        root / "outputs/sensitivity/comparison/selection_overlap.csv",
        root / "outputs/sensitivity/comparison/selection_summary.csv",
    )
    for path in csv_paths:
        require(path)
        summaries[path.relative_to(root).as_posix()] = pd.read_csv(path).to_dict(orient="records")
    return summaries


def build(root: Path, output_dir: Path, overwrite: bool) -> None:
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir == root or root not in output_dir.parents:
        raise ValueError("The bundle output must be a subdirectory of the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.iterdir())
    if existing and not overwrite:
        raise FileExistsError(f"Bundle directory is not empty: {output_dir}; pass --overwrite")
    if overwrite:
        unknown = [
            path
            for path in existing
            if path.name not in BUNDLE_FILENAMES
            and path.name not in {"main.tex", "recomputed_paper_tables.json", "reproducibility"}
        ]
        if unknown:
            raise ValueError(
                "Refusing to overwrite unexpected paper_artifacts entries: "
                + ", ".join(path.name for path in unknown)
            )
        for name in BUNDLE_FILENAMES:
            path = output_dir / name
            if path.is_file():
                path.unlink()

    outputs = root / "outputs"
    benchmark_path = require(outputs / "prep/benchmark.json")
    threshold_path = require(outputs / "prep/threshold_arcface.json")
    benchmark = read_json(benchmark_path)
    threshold = read_json(threshold_path)
    gallery_reference_path = require(outputs / "prep/gallery_reference_detections.csv")
    gallery_references = normalize_frame_paths(pd.read_csv(gallery_reference_path), root)

    group_features = pd.read_csv(require(outputs / "expanded_summary/expanded_mechanistic_group_features.csv"))
    experiment_names = sorted(group_features["experiment_name"].unique(), key=lambda x: int(x.split("_")[-1]))
    if len(experiment_names) != 30:
        raise ValueError(f"Expected 30 groups, found {len(experiment_names)}")

    run_metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    adapter_summaries: list[dict[str, Any]] = []
    generation_runs: list[dict[str, Any]] = []
    generation_attempts: list[dict[str, Any]] = []
    target_candidates: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    run_keys: list[str] = []
    adapter_arrays: dict[str, list[np.ndarray]] = {
        "forget_centroid": [],
        "target_embedding": [],
        "down": [],
        "up": [],
    }
    adapter_ranks: list[int] = []
    adapter_targets: list[int] = []

    selection_by_policy: dict[str, dict[str, dict[str, Any]]] = {}
    for policy in POLICIES:
        selection_path = require(outputs / f"expanded_target_selection_{policy}_summary.json")
        selection = read_json(selection_path)
        if len(selection["rows"]) != 30:
            raise ValueError(f"Expected 30 target rows for {policy}")
        selection_by_policy[policy] = {row["experiment_name"]: row for row in selection["rows"]}

    for experiment_name in experiment_names:
        forget_id = int(experiment_name.split("_")[-1])
        experiment_config = load_experiment_config(
            require(root / f"configs/expanded/{experiment_name}.json")
        )
        group = lookup_group(benchmark, forget_id)
        expected_keys = expected_generation_keys(experiment_config, group)
        for policy in POLICIES:
            run_key = f"{experiment_name}/{policy}"
            run_dir = outputs / experiment_name
            metrics_path = require(run_dir / f"eval_{policy}/metrics.json")
            predictions_path = require(run_dir / f"eval_{policy}/per_sample_results.csv")
            adapter_path = require(run_dir / f"u_projection_adapter_{policy}/projection_adapter.npz")
            adapter_summary_path = require(
                run_dir / f"u_projection_adapter_{policy}/projection_adapter_summary.json"
            )
            generation_manifest_path = require(run_dir / f"{policy}_gen/manifest.json")

            metrics = portable(read_json(metrics_path), root)
            metrics.pop("provenance", None)
            metrics = {
                "run_key": run_key,
                "experiment_name": experiment_name,
                "forget_id": forget_id,
                "policy": policy,
                **metrics,
            }
            if int(metrics["attempted_sample_count"]) != 168:
                raise ValueError(f"{run_key} has {metrics['attempted_sample_count']} attempts, expected 168")
            run_metrics.append(metrics)

            prediction_frame = pd.read_csv(predictions_path)
            prediction_frame.insert(0, "policy", policy)
            prediction_frame.insert(0, "forget_id", forget_id)
            prediction_frame.insert(0, "experiment_name", experiment_name)
            prediction_frame.insert(0, "run_key", run_key)
            prediction_frame = normalize_frame_paths(prediction_frame, root)
            if len(prediction_frame) != int(metrics["evaluated_sample_count"]):
                raise ValueError(f"Prediction count disagrees with metrics for {run_key}")
            predictions.append(prediction_frame)

            adapter_summary = portable(read_json(adapter_summary_path), root)
            adapter_summary.pop("provenance", None)
            adapter_summary = {
                "run_key": run_key,
                "experiment_name": experiment_name,
                "policy": policy,
                **adapter_summary,
            }
            if adapter_summary.get("checkpoint_selection_timing") != "pre_update_state_scored_and_saved_before_optimizer_step":
                raise ValueError(f"{run_key} does not use the required pre-update checkpoint timing")
            adapter_summaries.append(adapter_summary)

            with np.load(adapter_path) as adapter:
                for name in adapter_arrays:
                    adapter_arrays[name].append(np.asarray(adapter[name], dtype=np.float32))
                adapter_ranks.append(int(adapter["rank"]))
                adapter_targets.append(int(adapter["target_identity"]))

            generation_manifest = portable(read_json(generation_manifest_path), root)
            images = generation_manifest.pop("images")
            if len(images) != 168:
                raise ValueError(f"{run_key} generation manifest has {len(images)} images")
            if manifest_generation_keys({"images": images}) != expected_keys:
                raise ValueError(f"{run_key} generation manifest does not match its exact key plan")
            generation_runs.append(
                {
                    "run_key": run_key,
                    "experiment_name": experiment_name,
                    "policy": policy,
                    "variant": generation_manifest["variant"],
                    "forget_id": forget_id,
                    "retain_ids": experiment_config["retain_ids"],
                    "hard_retain_ids": experiment_config["hard_retain_ids"],
                    "random_retain_ids": experiment_config["random_retain_ids"],
                    "target_identity": int(generation_manifest["target_identity"]),
                    "rank": int(generation_manifest["rank"]),
                    "reference_det_thresh": float(generation_manifest["reference_det_thresh"]),
                    "generation_reference_det_thresh": float(
                        generation_manifest.get(
                            "generation_reference_det_thresh",
                            generation_manifest["reference_det_thresh"],
                        )
                    ),
                    "eval_seeds": experiment_config["eval_seeds"],
                    "generator": experiment_config["generator"],
                    "generation": experiment_config["generation"],
                    "attempt_count": len(images),
                }
            )
            detected_paths = set(prediction_frame["image_path"].astype(str))
            for image in images:
                image["run_key"] = run_key
                image["experiment_name"] = experiment_name
                image["forget_id"] = forget_id
                image["policy"] = policy
                image["evaluation_detector_success"] = image["image_path"] in detected_paths
                generation_attempts.append(image)

            target = portable(selection_by_policy[policy][experiment_name], root)
            target["run_key"] = run_key
            target["policy"] = policy
            if int(target["target_identity"]) != adapter_targets[-1]:
                raise ValueError(f"Target selection and adapter disagree for {run_key}")
            candidates = target.get("candidate_similarities", [])
            if not isinstance(candidates, list) or len(candidates) != 3:
                raise ValueError(f"{run_key} does not preserve all three target candidates")
            similarity_order = {
                int(candidate["identity"]): rank
                for rank, candidate in enumerate(
                    sorted(
                        candidates,
                        key=lambda item: (float(item["similarity"]), int(item["identity"])),
                    ),
                    start=1,
                )
            }
            for candidate_index, candidate in enumerate(candidates, start=1):
                candidate_identity = int(candidate["identity"])
                target_candidates.append(
                    {
                        "run_key": run_key,
                        "experiment_name": experiment_name,
                        "forget_id": forget_id,
                        "policy": policy,
                        "candidate_index": candidate_index,
                        "candidate_identity": candidate_identity,
                        "similarity_to_forget": float(candidate["similarity"]),
                        "similarity_rank_ascending": similarity_order[candidate_identity],
                        "selected": candidate_identity == int(target["target_identity"]),
                    }
                )
            target["candidate_similarities"] = json.dumps(
                candidates, sort_keys=True, separators=(",", ":")
            )
            target_rows.append(target)
            run_keys.append(run_key)

    metrics_frame = pd.DataFrame(run_metrics)
    metrics_frame.to_csv(output_dir / "run_metrics.csv", index=False, lineterminator="\n")
    predictions_frame = pd.concat(predictions, ignore_index=True)
    write_csv_gzip(predictions_frame, output_dir / "per_sample_predictions.csv.gz")
    write_jsonl(output_dir / "adapter_summaries.jsonl", adapter_summaries)
    write_jsonl(output_dir / "generation_runs.jsonl", generation_runs)
    generation_frame = normalize_frame_paths(pd.DataFrame(generation_attempts), root)
    write_csv_gzip(generation_frame, output_dir / "generation_attempts.csv.gz")
    pd.DataFrame(target_candidates).to_csv(
        output_dir / "target_candidates.csv", index=False, lineterminator="\n"
    )
    pd.DataFrame(target_rows).to_csv(output_dir / "target_selections.csv", index=False, lineterminator="\n")
    np.savez_compressed(
        output_dir / "adapters.npz",
        run_keys=np.asarray(run_keys),
        forget_centroid=np.stack(adapter_arrays["forget_centroid"]),
        target_embedding=np.stack(adapter_arrays["target_embedding"]),
        down=np.stack(adapter_arrays["down"]),
        up=np.stack(adapter_arrays["up"]),
        rank=np.asarray(adapter_ranks, dtype=np.int64),
        target_identity=np.asarray(adapter_targets, dtype=np.int64),
    )

    quality_root = outputs / "quality_official_v1"
    quality = pd.read_csv(require(quality_root / "generated_official_quality_per_image.csv"))
    real_quality = pd.read_csv(require(quality_root / "real_official_quality_per_image.csv"))
    if len(quality) != 20160 or quality["image_key"].nunique() != 20160:
        raise ValueError("Official generated quality rows are incomplete or duplicated")
    if len(real_quality) != 960 or real_quality["image_key"].nunique() != 960:
        raise ValueError("Official real-reference quality rows are incomplete or duplicated")
    if set(quality["status"]) != {"scored", "no_face"} or set(real_quality["status"]) != {"scored"}:
        raise ValueError("Official quality rows contain an unexpected status")
    quality["source"] = "generated"
    real_quality["source"] = "real_reference"
    quality = pd.concat([quality, real_quality], ignore_index=True)
    quality = normalize_frame_paths(quality, root)
    write_csv_gzip(quality, output_dir / "quality_scores.csv.gz")

    recognizer = normalize_frame_paths(
        pd.read_csv(require(outputs / "adaface_verification/adaface_vs_arcface_per_run.csv")), root
    )
    recognizer.to_csv(output_dir / "recognizer_runs.csv", index=False, lineterminator="\n")
    adaface_predictions = normalize_frame_paths(
        pd.read_csv(require(outputs / "adaface_verification/adaface_per_image.csv")), root
    )
    if (
        len(adaface_predictions) != 19901
        or adaface_predictions["run_key"].nunique() != 120
        or set(adaface_predictions["recognizer"]) != {"adaface"}
    ):
        raise ValueError("AdaFace per-image evidence is incomplete")
    write_csv_gzip(adaface_predictions, output_dir / "adaface_predictions.csv.gz")
    large_gallery = normalize_frame_paths(
        pd.read_csv(require(outputs / "large_gallery_leakage/large_gallery_per_image.csv")), root
    )
    write_csv_gzip(large_gallery, output_dir / "large_gallery_predictions.csv.gz")
    sensitivity = pd.read_csv(require(outputs / "sensitivity/comparison/policy_metrics.csv"))
    sensitivity.to_csv(output_dir / "sensitivity_policy_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "paper_table_values.tex").write_text(
        latex_constants(recompute(output_dir)), encoding="utf-8", newline="\n"
    )

    group_ids = {int(name.split("_")[-1]) for name in experiment_names}
    group_definitions = [
        portable(group, root) for group in benchmark["expanded_groups"] if int(group["forget_id"]) in group_ids
    ]
    if len(group_definitions) != 30:
        raise ValueError("The benchmark does not contain exactly the 30 evaluated group definitions")
    neighbor_path = require(outputs / "prep/identity_neighbors.csv")
    neighbors = pd.read_csv(neighbor_path)
    public_ids = {
        int(identity)
        for group in benchmark.get("public_groups", [])
        for identity in [group["forget_id"], *group.get("retain_ids", [])]
    }
    if neighbors["target_identity"].isin(public_ids).any() or neighbors[
        "retain_identity"
    ].isin(public_ids).any():
        raise ValueError("The saved neighbor graph still contains public identities")
    neighbor_counts = neighbors.groupby("target_identity").size()
    if (
        len(neighbors) != 595600
        or neighbors["target_identity"].nunique() != 5956
        or not (neighbor_counts == 100).all()
    ):
        raise ValueError("The non-public neighbor graph must contain 100 rows x 5,956 identities")
    hardness_top3 = neighbors.loc[neighbors["rank"] <= 3].copy()
    if len(hardness_top3) != 17868 or not (
        hardness_top3.groupby("target_identity").size() == 3
    ).all():
        raise ValueError("Every non-public identity must retain exactly three hardness neighbors")
    write_csv_gzip(hardness_top3, output_dir / "hardness_top3_neighbors.csv.gz")
    expected_gallery_references = {
        (int(identity), str(path))
        for group in group_definitions
        for identity, paths in group["eval_reference_images"].items()
        for path in paths
    }
    observed_gallery_references = set(
        gallery_references[["identity", "reference_image"]].itertuples(index=False, name=None)
    )
    if (
        len(gallery_references) != 480
        or gallery_references["identity"].nunique() != 160
        or not (gallery_references.groupby("identity").size() == 3).all()
        or len(observed_gallery_references) != 480
        or observed_gallery_references != expected_gallery_references
        or not gallery_references["detector_success"].astype(bool).all()
        or set(gallery_references["det_thresh"].astype(float)) != {0.1}
    ):
        raise ValueError("Held-out gallery-reference detection evidence is incomplete")
    gallery_references.to_csv(
        output_dir / "gallery_reference_detections.csv", index=False, lineterminator="\n"
    )
    quality_provenance_path = require(quality_root / "provenance.json")
    quality_comparison_path = require(quality_root / "comparison.json")
    quality_provenance = read_json(quality_provenance_path)
    protocol = {
        "scope": "30 split-consistent groups x 4 target policies",
        "policies": list(POLICIES),
        "target_policy_random_seed": 2026,
        "group_count": 30,
        "run_count": 120,
        "attempts_per_run": 168,
        "total_generation_attempts": 20160,
        "arcface_calibration": threshold,
        "reference_split_policy": benchmark.get("reference_split_policy"),
        "expanded_selection_policy": benchmark.get("expanded_selection_policy"),
        "centroid_construction_policy": benchmark.get("centroid_construction_policy"),
        "hardness_evidence_policy": {
            "eligible_non_public_identity_count": 5956,
            "neighbors_per_identity": 3,
            "excluded_public_identities": sorted(public_ids),
            "source_neighbor_depth": 100,
        },
        "detector_threshold_policy": {
            "fitting_conditioning_references": 0.5,
            "held_out_evaluation_pipeline": 0.1,
            "train_pool_and_calibration": "supplied CelebA landmarks; detector threshold not invoked",
        },
        "official_quality_protocol": {
            "protocol": quality_provenance["protocol"],
            "input_counts": quality_provenance["input_counts"],
        },
        "official_quality_aggregation": portable(read_json(quality_comparison_path), root)[
            "primary_aggregation"
        ],
        "expanded_groups": group_definitions,
    }
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "reported_summaries.json", collect_small_summaries(root))

    readme = r"""# Paper artifacts

These files are enough to inspect and recompute the published results without
rerunning adapter training or image generation. CelebA, generated images, and
pretrained model weights are not included.

Files in this directory:

- `run_metrics.csv` and `per_sample_predictions.csv.gz`: run-level metrics and
  the detected samples from which they are calculated.
- `generation_attempts.csv.gz`: all 20,160 attempted generations, including
  detection failures.
- `adapter_summaries.jsonl` and `adapters.npz`: the selected checkpoints for
  all 120 runs. `generation_runs.jsonl` records the generation settings without
  runner-specific orchestration fields.
- `target_candidates.csv`, `target_selections.csv`, and
  `hardness_top3_neighbors.csv.gz`: target-policy and neighbour evidence.
- `quality_scores.csv.gz`, `recognizer_runs.csv`, `adaface_predictions.csv.gz`,
  and `large_gallery_predictions.csv.gz`: additional evaluation evidence.
- `sensitivity_policy_metrics.csv`: the top-10/top-20 nested-cut experiment.
- `protocol.json`: exact groups, references, split, and calibration protocol.
- `reported_summaries.json` and `paper_table_values.tex`: derived summaries.

To rebuild these files and recompute the tables from completed experiment
outputs:

```powershell
venv\Scripts\python.exe scripts\reproducibility\build_paper_artifacts.py --overwrite
venv\Scripts\python.exe scripts\reproducibility\recompute_paper_tables.py `
  --json-output paper_artifacts\recomputed_paper_tables.json
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")

    file_count = sum(path.is_file() for path in output_dir.iterdir())
    print(f"Built {output_dir.relative_to(root)} with {file_count} files")
    print(f"Runs: {len(metrics_frame)}; attempts: {len(generation_frame)}; evaluated: {len(predictions_frame)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "paper_artifacts"
    build(root, output_dir, args.overwrite)


if __name__ == "__main__":
    main()
