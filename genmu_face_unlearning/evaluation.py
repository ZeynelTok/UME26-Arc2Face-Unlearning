from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .arcface_utils import ArcFaceEmbedder
from .benchmark import group_reference_map, load_benchmark, lookup_group
from .threshold import load_threshold_calibration


def _harmonic_mean(left: float, right: float) -> float:
    if left <= 0 or right <= 0:
        return 0.0
    return float(2.0 * left * right / (left + right))


def _collect_images(run_dir: Path) -> pd.DataFrame:
    rows = []
    for image_path in run_dir.rglob("*.png"):
        parts = image_path.relative_to(run_dir).parts
        if len(parts) < 3:
            continue
        rows.append(
            {
                "role": parts[0],
                "intended_identity": int(parts[1]),
                "seed": int(image_path.stem.split("_")[-1]),
                "image_path": str(image_path),
                "reference_image": parts[2],
            }
        )
    return pd.DataFrame(rows)


def _build_gallery(
    embedder: ArcFaceEmbedder,
    reference_images: dict[str, list[str]],
    expected_references_per_identity: int = 3,
) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    gallery = {}
    provenance: list[dict[str, object]] = []
    for identity_text, reference_paths in reference_images.items():
        if len(reference_paths) != expected_references_per_identity:
            raise RuntimeError(
                f"Gallery identity {identity_text} has {len(reference_paths)} references; "
                f"expected {expected_references_per_identity}"
            )
        embeddings = []
        for path in reference_paths:
            result = embedder.detect_and_embed_image(path)
            provenance.append(
                {
                    "identity": int(identity_text),
                    "reference_image": str(path),
                    "detector_success": result is not None,
                }
            )
            if result is None:
                raise RuntimeError(
                    f"Unable to detect required gallery reference for identity "
                    f"{identity_text}: {path}"
                )
            embeddings.append(result.embedding)
        centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise RuntimeError(f"Gallery centroid has zero norm for identity {identity_text}")
        gallery[int(identity_text)] = centroid / norm
    if set(gallery) != {int(identity) for identity in reference_images}:
        raise RuntimeError("Gallery identities differ from the required identity set")
    return gallery, provenance


def _aggregate_metrics(results: pd.DataFrame) -> dict[str, object]:
    forget_rows = results.loc[results["role"] == "forget"].copy()
    retain_rows = results.loc[results["role"] == "retain"].copy()
    fa = 100.0 * float(forget_rows["verifies_as_target"].mean()) if not forget_rows.empty else 0.0
    ea = 100.0 - fa
    ra = 100.0 * float(retain_rows["verifies_as_intended"].mean()) if not retain_rows.empty else 0.0
    erb = _harmonic_mean(ea, ra)

    metrics = {"FA": fa, "EA": ea, "RA": ra, "ERB": erb}
    grouped = forget_rows.groupby(["intended_identity", "reference_image"])
    for k in [1, 4, 8]:
        hits = []
        for (_, _), group in grouped:
            group = group.sort_values("seed").head(k)
            hits.append(bool(group["verifies_as_target"].any()))
        metrics[f"IdentityLeak@{k}"] = 100.0 * float(np.mean(hits)) if hits else 0.0

    metrics["mean_target_similarity"] = float(forget_rows["target_similarity"].mean()) if not forget_rows.empty else 0.0
    metrics["max_target_similarity"] = float(forget_rows["target_similarity"].max()) if not forget_rows.empty else 0.0

    hard_rows = retain_rows.loc[retain_rows["retain_group"] == "hard"].copy()
    random_rows = retain_rows.loc[retain_rows["retain_group"] == "random"].copy()
    hard_ra = 100.0 * float(hard_rows["verifies_as_intended"].mean()) if not hard_rows.empty else None
    random_ra = 100.0 * float(random_rows["verifies_as_intended"].mean()) if not random_rows.empty else None
    metrics["RA_hard"] = hard_ra
    metrics["RA_random"] = random_ra
    metrics["hard_neighbor_retention_gap"] = hard_ra - random_ra if hard_ra is not None and random_ra is not None else None
    metrics["forget_sample_count"] = int(len(forget_rows))
    metrics["retain_sample_count"] = int(len(retain_rows))
    metrics["total_sample_count"] = int(len(results))
    if "detector_success" in results.columns:
        metrics["detector_success_rate"] = 100.0 * float(results["detector_success"].mean()) if not results.empty else 0.0
        metrics["fallback_sample_count"] = int((~results["detector_success"]).sum()) if not results.empty else 0
    return metrics


def evaluate_run(
    config: dict,
    benchmark_path: Path | None,
    threshold_path: Path | None,
    after_dir: Path,
    arcface_model_root: Path | None = None,
    device: str = "cuda",
    det_thresh: float = 0.1,
    use_fallback: bool = False,
    embedder=None,
    threshold: dict | None = None,
    group: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Score generated images against the eval-reference gallery."""
    if group is None:
        if benchmark_path is None:
            raise ValueError("benchmark_path is required when group is not provided.")
        benchmark = load_benchmark(benchmark_path)
        group = lookup_group(benchmark, config["forget_id"])
    if threshold is None:
        if threshold_path is None:
            raise ValueError("threshold_path is required when threshold is not provided.")
        threshold = load_threshold_calibration(threshold_path)
    if embedder is None:
        embedder = ArcFaceEmbedder(
            model_root=arcface_model_root,
            device=device,
            det_thresh=det_thresh,
            use_fallback=use_fallback,
        )
    gallery, gallery_provenance = _build_gallery(
        embedder, group_reference_map(group, split="eval")
    )

    rows = _collect_images(after_dir)
    if rows.empty:
        raise RuntimeError(f"No generated images found under {after_dir}")

    hard_ids = set(config.get("hard_retain_ids") or config.get("retain_ids", []))
    results = []
    row_records = rows.to_dict("records")
    for row in tqdm(row_records, total=len(row_records), desc=f"eval:{after_dir.name}", unit="img", dynamic_ncols=True):
        detection = embedder.detect_and_embed_image(row["image_path"])
        if detection is None:
            continue
        similarities = {identity: embedder.cosine_similarity(detection.embedding, centroid) for identity, centroid in gallery.items()}
        target_similarity = similarities.get(config["forget_id"], -1.0)
        intended_similarity = similarities.get(row["intended_identity"], -1.0)
        predicted_identity = max(similarities, key=similarities.get)
        retain_group = "forget"
        if row["role"] == "retain":
            retain_group = "hard" if row["intended_identity"] in hard_ids else "random"
        results.append(
            {
                **row,
                "detector_success": bool(detection.detector_success),
                "target_similarity": float(target_similarity),
                "intended_similarity": float(intended_similarity),
                "predicted_identity": int(predicted_identity),
                "verifies_as_target": bool(target_similarity >= threshold["threshold"] and predicted_identity == config["forget_id"]),
                "verifies_as_intended": bool(intended_similarity >= threshold["threshold"] and predicted_identity == row["intended_identity"]),
                "retain_group": retain_group,
            }
        )

    result_frame = pd.DataFrame(results)
    metrics = _aggregate_metrics(result_frame)
    metrics["attempted_sample_count"] = int(len(rows))
    metrics["evaluated_sample_count"] = int(len(result_frame))
    metrics["dropped_no_face_count"] = int(len(rows) - len(result_frame))
    metrics["det_thresh"] = float(det_thresh)
    metrics["use_fallback"] = bool(use_fallback)
    metrics["after_dir"] = str(after_dir)
    metrics["threshold"] = threshold["threshold"]
    metrics["gallery_reference_count"] = len(gallery_provenance)
    metrics["gallery_identity_count"] = len(gallery)
    metrics["_gallery_reference_detections"] = gallery_provenance
    return result_frame, metrics


def write_evaluation_outputs(results: pd.DataFrame, metrics: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "per_sample_results.csv", index=False)
    gallery_rows = metrics.get("_gallery_reference_detections", [])
    if gallery_rows:
        pd.DataFrame(gallery_rows).to_csv(
            out_dir / "gallery_reference_detections.csv", index=False
        )
    serialized_metrics = {
        key: value for key, value in metrics.items() if key != "_gallery_reference_detections"
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(serialized_metrics, handle, indent=2)
