from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

import numpy as np


def build_identity_plan(config: dict) -> list[tuple[int, str]]:
    plan = [(int(config["forget_id"]), "forget")]
    plan.extend((int(identity), "retain") for identity in config["retain_ids"])
    plan.extend((int(identity), "retain") for identity in config.get("random_retain_ids", []))
    return plan


def reference_face_embedding(embedder, reference_path: str) -> np.ndarray:
    detection = embedder.detect_and_embed_image(reference_path)
    if detection is None:
        raise RuntimeError(f"Unable to detect face in reference image: {reference_path}")
    return detection.embedding


def save_generated_images(
    *,
    runtime,
    out_dir: Path,
    role: str,
    identity: int,
    reference_path: str,
    face_embedding: np.ndarray,
    seeds: list[int],
    settings: dict,
) -> list[dict]:
    output_dir = out_dir / role / str(identity) / Path(reference_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for seed, image in runtime.generate_images(face_embedding, seeds=seeds, settings=settings):
        image_path = output_dir / f"seed_{seed}.png"
        image.save(image_path)
        records.append(
            {
                "role": role,
                "identity": int(identity),
                "reference_image": reference_path,
                "seed": int(seed),
                "image_path": str(image_path),
            }
        )
    return records


def generate_identity_plan_images(
    *,
    runtime,
    embedder,
    out_dir: Path,
    config: dict,
    reference_lookup: Callable[[int], list[str]],
    transform: Callable[[int, str, str, np.ndarray], tuple[np.ndarray, dict]] | None = None,
) -> list[dict]:
    records = []
    for identity, role in build_identity_plan(config):
        for reference_path in reference_lookup(identity):
            face_embedding = reference_face_embedding(embedder, reference_path)
            extra = {}
            if transform is not None:
                face_embedding, extra = transform(identity, role, reference_path, face_embedding)
            for record in save_generated_images(
                runtime=runtime,
                out_dir=out_dir,
                role=role,
                identity=identity,
                reference_path=reference_path,
                face_embedding=face_embedding,
                seeds=config["eval_seeds"],
                settings=config["generation"],
            ):
                record.update(extra)
                records.append(record)
    return records


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
