from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EXPECTED_IMAGE_COUNT = 202_599
EXPECTED_IDENTITY_COUNT = 10_177

LANDMARK_COLUMNS = [
    "lefteye_x",
    "lefteye_y",
    "righteye_x",
    "righteye_y",
    "nose_x",
    "nose_y",
    "leftmouth_x",
    "leftmouth_y",
    "rightmouth_x",
    "rightmouth_y",
]


@dataclass
class CelebAMetadataPaths:
    data_dir: Path

    @property
    def anno_dir(self) -> Path:
        return self.data_dir / "Anno"

    @property
    def image_dir(self) -> Path:
        return self.data_dir / "img_align_celeba"


def _read_identity_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_name, identity = line.strip().split()
            rows.append({"image_name": image_name, "identity": int(identity)})
    return pd.DataFrame(rows)


def _read_partition_file(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            image_name, partition = line.strip().split()
            rows.append({"image_name": image_name, "partition": int(partition)})
    return pd.DataFrame(rows)


def _read_landmarks_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=2,
        names=["image_name", *LANDMARK_COLUMNS],
    )


def build_metadata(data_dir: Path) -> pd.DataFrame:
    paths = CelebAMetadataPaths(data_dir)
    identities = _read_identity_file(paths.anno_dir / "identity_CelebA.txt")
    partitions = _read_partition_file(paths.data_dir / "list_eval_partition.txt")
    landmarks = _read_landmarks_file(paths.anno_dir / "list_landmarks_align_celeba.txt")
    metadata = identities.merge(partitions, on="image_name", how="inner", validate="one_to_one")
    metadata = metadata.merge(landmarks, on="image_name", how="inner", validate="one_to_one")
    metadata["image_path"] = metadata["image_name"].map(lambda name: str((paths.image_dir / name).resolve()))
    if len(metadata) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Incomplete CelebA annotations: expected {EXPECTED_IMAGE_COUNT:,} images, "
            f"found {len(metadata):,}."
        )
    identity_count = int(metadata["identity"].nunique())
    if identity_count != EXPECTED_IDENTITY_COUNT:
        raise ValueError(
            f"Incomplete CelebA identities: expected {EXPECTED_IDENTITY_COUNT:,}, "
            f"found {identity_count:,}."
        )
    missing = [path for path in metadata["image_path"] if not Path(path).is_file()]
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(
            f"CelebA is missing {len(missing):,} aligned images; first missing paths: {preview}"
        )
    return metadata


def split_reference_images(
    metadata: pd.DataFrame,
    identity: int,
    fit_count: int = 3,
    eval_count: int = 3,
) -> dict[str, list[str]]:
    identity_rows = metadata.loc[metadata["identity"] == identity].copy()
    identity_rows = identity_rows.sort_values(["partition", "image_name"]).reset_index(drop=True)
    if identity_rows.empty:
        return {"fit": [], "eval": []}

    fit_paths = identity_rows["image_path"].head(fit_count).tolist()
    eval_paths = identity_rows["image_path"].iloc[fit_count : fit_count + eval_count].tolist()

    if len(eval_paths) < eval_count:
        raise ValueError(
            f"Identity {identity} does not have enough disjoint images for "
            f"{fit_count} fit refs and {eval_count} eval refs."
        )
    return {"fit": fit_paths, "eval": eval_paths}
