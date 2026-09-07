from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .face_embedding import normalize_embedding


def _benchmark_eval_reference_paths(benchmark: dict) -> set[str]:
    paths = set()
    for refs in benchmark.get("global_eval_reference_images", {}).values():
        paths.update(str(path) for path in refs)
    return paths


def centroid(embeddings: list[np.ndarray] | np.ndarray) -> np.ndarray:
    matrix = np.stack(list(embeddings), axis=0) if isinstance(embeddings, list) else embeddings
    return normalize_embedding(matrix.mean(axis=0))


def select_identity_embedding(
    forget_embedding: np.ndarray,
    forget_id: int,
    benchmark: dict,
    metadata_path: Path,
    embeddings_path: Path,
    mode: str,
    candidate_ids: list[int] | None = None,
) -> tuple[int, np.ndarray, float]:
    metadata = pd.read_csv(metadata_path, usecols=["identity", "image_path"])
    embeddings = np.load(embeddings_path)
    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embedding row counts differ.")

    eval_reference_paths = _benchmark_eval_reference_paths(benchmark)
    if eval_reference_paths:
        keep = ~metadata["image_path"].isin(eval_reference_paths)
        metadata = metadata.loc[keep].reset_index(drop=True)
        embeddings = embeddings[keep.to_numpy()]

    if candidate_ids:
        allowed = {int(identity) for identity in candidate_ids}
    else:
        allowed = {int(identity) for identity in benchmark.get("global_reference_images", {}).keys()}
        if not allowed:
            allowed = {int(identity) for identity in metadata["identity"].unique()}

    identity_centroids = {}
    for identity, group in metadata.groupby("identity", sort=False):
        identity = int(identity)
        if identity not in allowed:
            continue
        identity_centroids[identity] = centroid(embeddings[group.index.to_numpy()])

    best_identity = None
    best_similarity = float("inf") if mode == "dissimilar" else -float("inf")
    for identity, candidate in identity_centroids.items():
        if identity == forget_id:
            continue
        similarity = float(np.dot(forget_embedding, candidate))
        if mode == "dissimilar" and similarity < best_similarity:
            best_identity = identity
            best_similarity = similarity
        elif mode == "nearest" and similarity > best_similarity:
            best_identity = identity
            best_similarity = similarity

    if best_identity is None:
        raise RuntimeError(f"Unable to find a {mode} benchmark identity.")
    return int(best_identity), identity_centroids[best_identity], best_similarity


def apply_projection_adapter(embedding: np.ndarray, adapter: dict) -> tuple[np.ndarray, float, float]:
    embedding = normalize_embedding(embedding)
    down = adapter["down"].astype(np.float32)
    up = adapter["up"].astype(np.float32)
    residual = up @ (down @ embedding)
    transformed = normalize_embedding(embedding + residual)
    cosine_to_forget = float(np.dot(embedding, adapter["forget_centroid"]))
    residual_norm = float(np.linalg.norm(residual))
    return transformed, cosine_to_forget, residual_norm


def load_projection_adapter(path: Path) -> dict:
    data = np.load(path, allow_pickle=False)
    return {
        "forget_centroid": normalize_embedding(data["forget_centroid"].astype(np.float32)),
        "target_embedding": normalize_embedding(data["target_embedding"].astype(np.float32)),
        "down": data["down"].astype(np.float32),
        "up": data["up"].astype(np.float32),
        "rank": int(data["rank"]),
        "target_identity": int(data["target_identity"]),
    }
