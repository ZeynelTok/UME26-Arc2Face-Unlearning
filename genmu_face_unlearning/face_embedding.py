"""Small shared helpers for ArcFace and AdaFace embedding code."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass
class DetectionResult:
    embedding: np.ndarray
    landmarks: np.ndarray | None = None
    bbox: np.ndarray | None = None
    detector_success: bool = True


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = embedding.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize_embedding(left), normalize_embedding(right)))


def landmarks_from_row(row: pd.Series) -> np.ndarray:
    return np.array(
        [
            [row["lefteye_x"], row["lefteye_y"]],
            [row["righteye_x"], row["righteye_y"]],
            [row["nose_x"], row["nose_y"]],
            [row["leftmouth_x"], row["leftmouth_y"]],
            [row["rightmouth_x"], row["rightmouth_y"]],
        ],
        dtype=np.float32,
    )


def largest_face(faces: list) -> dict:
    return sorted(
        faces,
        key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]),
    )[-1]


def extract_real_image_embeddings(metadata: pd.DataFrame, embedder, desc: str) -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    embeddings = []
    for row in tqdm(metadata.to_dict("records"), total=len(metadata), desc=desc):
        result = embedder.embed_aligned_row(pd.Series(row))
        rows.append(row)
        embeddings.append(result.embedding)
    if not embeddings:
        raise RuntimeError("No real-image embeddings were extracted.")
    return pd.DataFrame(rows), np.stack(embeddings, axis=0)
