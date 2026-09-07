"""Build and validate the generation records for an experiment."""

from __future__ import annotations

from pathlib import Path

from .benchmark import group_reference_list
from .generation import build_identity_plan


def expected_generation_keys(config: dict, group: dict) -> list[dict[str, object]]:
    keys: list[dict[str, object]] = []
    for identity, role in build_identity_plan(config):
        for reference_path in group_reference_list(group, identity, split="fit"):
            reference_stem = Path(reference_path).stem
            for seed in config["eval_seeds"]:
                relative_path = Path(role) / str(identity) / reference_stem / f"seed_{int(seed)}.png"
                keys.append(
                    {
                        "role": role,
                        "identity": int(identity),
                        "reference_stem": reference_stem,
                        "seed": int(seed),
                        "relative_image_path": relative_path.as_posix(),
                    }
                )
    return keys


def manifest_generation_keys(manifest: dict) -> list[dict[str, object]]:
    keys = []
    for image in manifest.get("images", []):
        role = str(image["role"])
        identity = int(image["identity"])
        reference_stem = Path(str(image["reference_image"])).stem
        seed = int(image["seed"])
        keys.append(
            {
                "role": role,
                "identity": identity,
                "reference_stem": reference_stem,
                "seed": seed,
                "relative_image_path": (
                    Path(role) / str(identity) / reference_stem / f"seed_{seed}.png"
                ).as_posix(),
            }
        )
    return keys
