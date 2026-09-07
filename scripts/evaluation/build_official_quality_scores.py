"""Score generated and reference images with SER-FIQ and GraFIQs.

Both methods use the largest SCRFD detection at threshold 0.1, aligned to
112x112. Every attempted input is recorded, including detection and scoring
failures, and the output can be resumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.benchmark import iter_groups, load_benchmark
from genmu_face_unlearning.evaluation import _collect_images
from genmu_face_unlearning.face_embedding import largest_face
from genmu_face_unlearning.official_quality import (
    OfficialGraFIQsScorer,
    OfficialSerFiqOnnxScorer,
)

POLICIES = ["nearest_hard", "random_hard", "median_hard", "least_sim_hard"]
SER_FIQ_COMMIT = "611296605db57b8d50518fd5911d5111eeb52747"
GRAFIQS_COMMIT = "8d11a1f8506fb0f86ebd7738afc31467e2a01058"

COMMON_FIELDS = [
    "image_key",
    "image_path",
    "source",
    "experiment_name",
    "forget_id",
    "policy",
    "role",
    "intended_identity",
    "reference_image",
    "seed",
    "real_identity",
    "real_splits",
    "real_group_ids",
    "input_bytes",
    "status",
    "error",
    "face_count",
    "selected_det_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "ser_fiq_num_passes",
    "ser_fiq_seed",
    "ser_fiq_execution",
    "ser_fiq_mean_euclidean_distance",
    "ser_fiq_paper_score",
    "ser_fiq_normalized_score",
    "grafiqs_image",
    "grafiqs_block1",
    "grafiqs_block2",
    "grafiqs_block3",
    "grafiqs_block4",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_image_key(path: Path) -> str:
    """Use a location-independent key when an input is inside the repository."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _discover_group_ids(out_root: Path) -> list[int]:
    ids = []
    for group_dir in out_root.glob("expanded_*"):
        suffix = group_dir.name.removeprefix("expanded_")
        if suffix.isdigit() and all((group_dir / f"{policy}_gen").is_dir() for policy in POLICIES):
            ids.append(int(suffix))
    return sorted(ids)


def _generated_records(out_root: Path, group_ids: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for group_id in group_ids:
        experiment_name = f"expanded_{group_id}"
        for policy in POLICIES:
            run_dir = out_root / experiment_name / f"{policy}_gen"
            images = _collect_images(run_dir)
            if len(images) != 168:
                raise RuntimeError(f"Expected 168 images in {run_dir}, found {len(images)}")
            images = images.sort_values(
                ["role", "intended_identity", "reference_image", "seed", "image_path"],
                kind="stable",
            )
            for record in images.to_dict("records"):
                path = Path(record["image_path"])
                records.append(
                    {
                        "image_key": _stable_image_key(path),
                        "image_path": str(path),
                        "source": "generated",
                        "experiment_name": experiment_name,
                        "forget_id": group_id,
                        "policy": policy,
                        "role": record["role"],
                        "intended_identity": int(record["intended_identity"]),
                        "reference_image": str(record["reference_image"]),
                        "seed": int(record["seed"]),
                    }
                )
    if len(records) != 20_160:
        raise RuntimeError(f"Expected 20,160 generated images, found {len(records)}")
    return records


def _real_records(benchmark_path: Path, group_ids: list[int]) -> list[dict[str, Any]]:
    groups = {int(group["forget_id"]): group for group in iter_groups(load_benchmark(benchmark_path))}
    missing = sorted(set(group_ids) - set(groups))
    if missing:
        raise RuntimeError(f"Benchmark is missing generated group IDs: {missing}")

    memberships: dict[str, dict[str, Any]] = {}
    for group_id in group_ids:
        group = groups[group_id]
        for split, field in (("fit", "fit_reference_images"), ("eval", "eval_reference_images")):
            for identity_text, paths in group[field].items():
                for path_text in paths:
                    path = Path(path_text)
                    key = _stable_image_key(path)
                    entry = memberships.setdefault(
                        key,
                        {
                            "image_key": key,
                            "image_path": str(path),
                            "source": "real",
                            "real_identity": int(identity_text),
                            "splits": set(),
                            "group_ids": set(),
                        },
                    )
                    if entry["real_identity"] != int(identity_text):
                        raise RuntimeError(f"Real image assigned to multiple identities: {path}")
                    entry["splits"].add(split)
                    entry["group_ids"].add(group_id)

    records = []
    for key in sorted(memberships):
        item = memberships[key]
        records.append(
            {
                "image_key": item["image_key"],
                "image_path": item["image_path"],
                "source": "real",
                "real_identity": item["real_identity"],
                "real_splits": json.dumps(sorted(item["splits"]), separators=(",", ":")),
                "real_group_ids": json.dumps(sorted(item["group_ids"]), separators=(",", ":")),
            }
        )
    if len(records) != 960:
        raise RuntimeError(f"Expected 960 unique real references, found {len(records)}")
    return records


def _pilot(records: list[dict[str, Any]], count: int, field: str | None = None) -> list[dict[str, Any]]:
    if count <= 0:
        return records
    if field is None:
        if count >= len(records):
            return records
        indices = np.linspace(0, len(records) - 1, count, dtype=int)
        return [records[int(index)] for index in indices]

    chosen = []
    values = sorted({str(record[field]) for record in records})
    for value in values:
        subset = [record for record in records if str(record[field]) == value]
        chosen.extend(_pilot(subset, count))
    return chosen


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=["image_key"], keep_default_na=False)
    if frame["image_key"].duplicated().any():
        raise RuntimeError(f"Duplicate image keys in resumable output: {path}")
    return set(frame["image_key"].astype(str))


def _score_record(
    record: dict[str, Any],
    detector: ArcFaceEmbedder,
    ser_fiq: OfficialSerFiqOnnxScorer,
    grafiqs: OfficialGraFIQsScorer,
) -> dict[str, Any]:
    row = {field: "" for field in COMMON_FIELDS}
    row.update(record)
    path = Path(str(record["image_path"]))
    try:
        row["input_bytes"] = path.stat().st_size
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("cv2.imread returned None")
        faces = detector.app.get(image)
        row["face_count"] = len(faces)
        if not faces:
            row["status"] = "no_face"
            return row
        face = largest_face(faces)
        bbox = np.asarray(face["bbox"], dtype=np.float64)
        row.update(
            {
                "selected_det_score": float(face["det_score"]),
                "bbox_x1": float(bbox[0]),
                "bbox_y1": float(bbox[1]),
                "bbox_x2": float(bbox[2]),
                "bbox_y2": float(bbox[3]),
            }
        )
        aligned = detector._norm_crop(image, landmark=np.asarray(face["kps"], dtype=np.float32))
        row.update(ser_fiq.score_aligned_bgr(aligned, image_key=str(record["image_key"])))
        row.update(grafiqs.score_aligned_bgr(aligned))
        row["status"] = "scored"
    except Exception as exc:  # Record failures and continue the long run.
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _score_records(
    records: list[dict[str, Any]],
    output_path: Path,
    detector: ArcFaceEmbedder,
    ser_fiq: OfficialSerFiqOnnxScorer,
    grafiqs: OfficialGraFIQsScorer,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys(output_path)
    pending = [record for record in records if str(record["image_key"]) not in completed]
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMMON_FIELDS, extrasaction="raise")
        if mode == "w":
            writer.writeheader()
            handle.flush()
        for record in tqdm(pending, desc=output_path.stem, unit="img", dynamic_ncols=True):
            writer.writerow(_score_record(record, detector, ser_fiq, grafiqs))
            handle.flush()


def _summarize(output_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(output_path, keep_default_na=False)
    scored = frame.loc[frame["status"] == "scored"].copy()
    for column in (
        "ser_fiq_normalized_score",
        "ser_fiq_paper_score",
        "grafiqs_image",
        "grafiqs_block2",
    ):
        scored[column] = pd.to_numeric(scored[column], errors="raise")
    summary: dict[str, Any] = {
        "n_attempted": int(len(frame)),
        "n_scored": int(len(scored)),
        "status_counts": {str(k): int(v) for k, v in frame["status"].value_counts().items()},
        "ser_fiq_normalized_mean": float(scored["ser_fiq_normalized_score"].mean()),
        "ser_fiq_normalized_std": float(scored["ser_fiq_normalized_score"].std(ddof=1)),
        "ser_fiq_paper_mean": float(scored["ser_fiq_paper_score"].mean()),
        "grafiqs_primary": "grafiqs_block2",
        "grafiqs_block2_mean": float(scored["grafiqs_block2"].mean()),
        "grafiqs_block2_std": float(scored["grafiqs_block2"].std(ddof=1)),
    }
    if "policy" in scored and (scored["policy"] != "").any():
        summary["per_policy"] = {}
        for policy in POLICIES:
            subset = scored.loc[scored["policy"] == policy]
            summary["per_policy"][policy] = {
                "n_scored": int(len(subset)),
                "ser_fiq_normalized_mean": float(subset["ser_fiq_normalized_score"].mean()),
                "ser_fiq_normalized_std": float(subset["ser_fiq_normalized_score"].std(ddof=1)),
                "grafiqs_block2_mean": float(subset["grafiqs_block2"].mean()),
                "grafiqs_block2_std": float(subset["grafiqs_block2"].std(ddof=1)),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["generated", "real", "all"], default="all")
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "outputs" / "prep" / "benchmark.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "quality_official_v1")
    parser.add_argument("--insightface-model-root", type=Path, default=ROOT / "models" / "insightface")
    parser.add_argument(
        "--ser-fiq-model",
        type=Path,
        default=ROOT / "models" / "official_quality" / "ser_fiq_arcface_t100_mask.onnx",
    )
    parser.add_argument("--ser-fiq-trunk-model", type=Path, default=None)
    parser.add_argument("--ser-fiq-head-model", type=Path, default=None)
    parser.add_argument(
        "--grafiqs-source",
        type=Path,
        default=ROOT / "models" / "official_quality" / "grafiqs_source",
    )
    parser.add_argument(
        "--grafiqs-weights",
        type=Path,
        default=ROOT / "models" / "official_quality" / "resnet100_ms1mv2_arcface.pth",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--det-thresh", type=float, default=0.1)
    parser.add_argument("--base-seed", type=int, default=20260810)
    parser.add_argument("--pilot-generated-per-policy", type=int, default=0)
    parser.add_argument("--pilot-real", type=int, default=0)
    args = parser.parse_args()
    if args.det_thresh != 0.1:
        raise ValueError("The paper protocol uses --det-thresh 0.1")

    group_ids = _discover_group_ids(args.out_root)
    if len(group_ids) != 30:
        raise RuntimeError(f"Expected 30 complete generated groups, found {len(group_ids)}")
    generated = _generated_records(args.out_root, group_ids)
    real = _real_records(args.benchmark, group_ids)
    generated = _pilot(generated, args.pilot_generated_per_policy, field="policy")
    real = _pilot(real, args.pilot_real)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = output_dir / "provenance.json"
    model_hashes = {
        "ser_fiq_onnx": _sha256(args.ser_fiq_model),
        "grafiqs_weights": _sha256(args.grafiqs_weights),
    }
    if args.ser_fiq_trunk_model is not None and args.ser_fiq_head_model is not None:
        model_hashes["ser_fiq_trunk_onnx"] = _sha256(args.ser_fiq_trunk_model)
        model_hashes["ser_fiq_head_onnx"] = _sha256(args.ser_fiq_head_model)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "protocol": {
            "detector": "InsightFace antelopev2 SCRFD; largest bounding-box area",
            "det_thresh": args.det_thresh,
            "alignment": "InsightFace five-landmark norm_crop, 112x112",
            "ser_fiq": (
                "official ArcFace; Dropout(0.5); T=100; alpha=130; r=0.88; "
                + (
                    "deterministic trunk once, stochastic final Dropout/head repeated"
                    if args.ser_fiq_trunk_model is not None
                    else "full network repeated"
                )
            ),
            "grafiqs": (
                "official ArcFace ResNet-100; B2 gradient primary as selected in the paper's "
                "SOTA comparison; image/B1/B3/B4 retained; BGR input"
            ),
        },
        "source_commits": {"ser_fiq": SER_FIQ_COMMIT, "grafiqs": GRAFIQS_COMMIT},
        "model_sha256": model_hashes,
        "ser_fiq_seed_key": "repository-relative POSIX image_key stored in each CSV row",
        "base_seed": args.base_seed,
        "input_counts": {"generated": len(generated), "real_unique": len(real)},
        "commands": " ".join(sys.argv),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")

    detector = ArcFaceEmbedder(
        model_root=args.insightface_model_root,
        device=args.device,
        det_thresh=args.det_thresh,
        use_fallback=False,
    )
    ser_fiq = OfficialSerFiqOnnxScorer(
        args.ser_fiq_model,
        trunk_model_path=args.ser_fiq_trunk_model,
        head_model_path=args.ser_fiq_head_model,
        device=args.device,
        base_seed=args.base_seed,
    )
    grafiqs = OfficialGraFIQsScorer(
        args.grafiqs_source,
        args.grafiqs_weights,
        backbone="iresnet100",
        device=args.device,
        bgr2rgb=False,
    )

    outputs = []
    if args.scope in {"generated", "all"}:
        path = output_dir / "generated_official_quality_per_image.csv"
        _score_records(generated, path, detector, ser_fiq, grafiqs)
        outputs.append(path)
    if args.scope in {"real", "all"}:
        path = output_dir / "real_official_quality_per_image.csv"
        _score_records(real, path, detector, ser_fiq, grafiqs)
        outputs.append(path)

    summaries = {path.stem: _summarize(path) for path in outputs}
    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
