from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def cosine_scores(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left * right, axis=1)


def calibrate_arcface_threshold(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    far_target: float = 0.01,
    same_pairs_per_identity: int = 4,
    diff_pairs_per_identity: int = 4,
    name: str = "arcface_far_0p01",
) -> dict:
    rng = np.random.default_rng(1234)
    grouped = metadata.groupby("identity").indices
    same_left = []
    same_right = []
    diff_left = []
    diff_right = []

    identities = [identity for identity, indices in grouped.items() if len(indices) >= 2]
    all_identities = list(grouped.keys())

    for identity in identities:
        indices = np.array(grouped[identity])
        for _ in range(min(same_pairs_per_identity, len(indices) // 2 + 1)):
            pair = rng.choice(indices, size=2, replace=False)
            same_left.append(embeddings[pair[0]])
            same_right.append(embeddings[pair[1]])
        other_identities = [x for x in all_identities if x != identity]
        if not other_identities:
            continue
        for _ in range(diff_pairs_per_identity):
            left_index = int(rng.choice(indices))
            other_identity = int(rng.choice(other_identities))
            right_index = int(rng.choice(grouped[other_identity]))
            diff_left.append(embeddings[left_index])
            diff_right.append(embeddings[right_index])

    same_scores = cosine_scores(np.stack(same_left, axis=0), np.stack(same_right, axis=0))
    diff_scores = cosine_scores(np.stack(diff_left, axis=0), np.stack(diff_right, axis=0))

    thresholds = np.sort(np.unique(np.concatenate([same_scores, diff_scores])))
    best_threshold = float(thresholds[-1])
    best_far = 1.0
    for threshold in thresholds:
        far = float((diff_scores >= threshold).mean())
        if far <= far_target:
            best_threshold = float(threshold)
            best_far = far
            break

    return {
        "name": name,
        "threshold": best_threshold,
        "far_target": far_target,
        "false_accept_rate": best_far,
        "num_same_pairs": int(len(same_scores)),
        "num_diff_pairs": int(len(diff_scores)),
    }


def save_threshold_calibration(calibration: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)


def load_threshold_calibration(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
