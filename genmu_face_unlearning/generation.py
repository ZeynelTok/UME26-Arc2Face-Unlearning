from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_identity_plan(config: dict) -> list[tuple[int, str]]:
    plan = [(int(config["forget_id"]), "forget")]
    plan.extend((int(identity), "retain") for identity in config["retain_ids"])
    plan.extend((int(identity), "retain") for identity in config.get("random_retain_ids", []))
    return plan


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


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
