from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .celeba import split_reference_images


def load_public_face_groups(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    groups = []
    for item in payload.get("splits", []):
        if str(item.get("track", "")).lower() != "face":
            continue
        groups.append(
            {
                "set_name": str(item["set"]),
                "forget_id": int(item["forget_id"]),
                "retain_ids": [int(x) for x in item.get("retain_ids", [])],
                "selection_note": str(item.get("selection_note", "Organizer-provided public validation group")),
            }
        )
    if not groups:
        raise ValueError(f"No face splits were found in {path}")
    return groups


def build_identity_centroids(metadata: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    rows = []
    for identity, group in metadata.groupby("identity", sort=True):
        indices = group.index.to_numpy()
        centroid = embeddings[indices].mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)
        rows.append(
            {
                "identity": int(identity),
                "num_images": int(len(group)),
                "centroid": centroid,
                "partition": int(group["partition"].mode().iat[0]),
            }
        )
    return pd.DataFrame(rows)


def compute_identity_neighbors(centroids: pd.DataFrame, top_k: int = 50) -> pd.DataFrame:
    centroid_matrix = np.stack(centroids["centroid"].to_list(), axis=0)
    distances = 1.0 - (centroid_matrix @ centroid_matrix.T)
    rows = []
    identities = centroids["identity"].tolist()
    counts = centroids["num_images"].tolist()

    for i, target_identity in enumerate(identities):
        order = np.argsort(distances[i])
        rank = 0
        for j in order:
            if i == j:
                continue
            rank += 1
            rows.append(
                {
                    "target_identity": int(target_identity),
                    "retain_identity": int(identities[j]),
                    "target_num_images": int(counts[i]),
                    "retain_num_images": int(counts[j]),
                    "cosine_distance": float(distances[i, j]),
                    "rank": rank,
                }
            )
            if rank >= top_k:
                break
    return pd.DataFrame(rows)


def build_global_reference_maps(
    metadata: pd.DataFrame,
    identities: list[int],
    fit_reference_count: int,
    eval_reference_count: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Create the deterministic global reference split before benchmark selection."""
    fit_reference_images: dict[str, list[str]] = {}
    eval_reference_images: dict[str, list[str]] = {}
    for identity in sorted(int(value) for value in identities):
        refs = split_reference_images(
            metadata,
            identity,
            fit_count=fit_reference_count,
            eval_count=eval_reference_count,
        )
        fit_reference_images[str(identity)] = refs["fit"]
        eval_reference_images[str(identity)] = refs["eval"]
    return fit_reference_images, eval_reference_images


def _sample_random_retains(
    rng: np.random.Generator,
    candidate_identities: list[int],
    forbidden: set[int],
    count: int,
    excluded: set[int] | None = None,
) -> list[int]:
    """Sample uniformly while preserving valid draws from the seeded sequence."""
    excluded = excluded or set()
    pool = [identity for identity in candidate_identities if identity not in forbidden]
    eligible_pool = [identity for identity in pool if identity not in excluded]
    if len(eligible_pool) < count:
        return eligible_pool
    initial = [int(x) for x in rng.choice(pool, size=count, replace=False)]
    selected = [identity for identity in initial if identity not in excluded]
    if len(selected) < count:
        remaining = [identity for identity in eligible_pool if identity not in selected]
        selected.extend(
            int(x)
            for x in rng.choice(remaining, size=count - len(selected), replace=False)
        )
    return selected


def build_benchmark(
    metadata: pd.DataFrame,
    centroids: pd.DataFrame,
    neighbors: pd.DataFrame,
    min_images_per_identity: int,
    public_face_groups: list[dict],
    expanded_forget_count: int = 20,
    hard_retain_count: int = 3,
    fit_reference_count: int = 3,
    eval_reference_count: int = 3,
) -> dict:
    rng = np.random.default_rng(2026)
    eligible = centroids.loc[centroids["num_images"] >= min_images_per_identity].copy()
    public_ids = {group["forget_id"] for group in public_face_groups}
    public_ids.update(identity for group in public_face_groups for identity in group["retain_ids"])
    non_public_eligible = eligible.loc[~eligible["identity"].isin(public_ids)].copy()
    non_public_eligible_ids = set(non_public_eligible["identity"].astype(int).tolist())

    public_groups = []
    for group in public_face_groups:
        all_ids = [group["forget_id"], *group["retain_ids"]]
        fit_reference_images = {}
        eval_reference_images = {}
        for identity in all_ids:
            refs = split_reference_images(
                metadata,
                identity,
                fit_count=fit_reference_count,
                eval_count=eval_reference_count,
            )
            fit_reference_images[str(identity)] = refs["fit"]
            eval_reference_images[str(identity)] = refs["eval"]
        public_groups.append(
            {
                "set_name": group["set_name"],
                "forget_id": group["forget_id"],
                "retain_ids": list(group["retain_ids"]),
                "hard_retain_ids": list(group["retain_ids"]),
                "random_retain_ids": [],
                "fit_reference_images": fit_reference_images,
                "eval_reference_images": eval_reference_images,
                "selection_note": group.get("selection_note", "Organizer-provided public validation group"),
            }
        )

    candidate_neighbors = neighbors.loc[
        neighbors["target_identity"].isin(non_public_eligible_ids)
        & neighbors["retain_identity"].isin(non_public_eligible_ids)
    ].copy()
    candidate_neighbors = candidate_neighbors.sort_values(
        ["target_identity", "cosine_distance", "retain_identity"], kind="mergesort"
    )
    candidate_neighbors["rank"] = (
        candidate_neighbors.groupby("target_identity", sort=False).cumcount() + 1
    )
    neighbor_counts = candidate_neighbors.groupby("target_identity").size()
    complete_targets = set(
        neighbor_counts.loc[neighbor_counts >= hard_retain_count].index.astype(int).tolist()
    )
    candidate_neighbors = candidate_neighbors.loc[
        candidate_neighbors["target_identity"].isin(complete_targets)
    ].copy()

    hardness = (
        candidate_neighbors.loc[candidate_neighbors["rank"] <= hard_retain_count]
        .groupby("target_identity", as_index=False)["cosine_distance"]
        .mean()
        .rename(columns={"cosine_distance": "mean_topk_distance"})
        .sort_values(["mean_topk_distance", "target_identity"], ascending=[True, True])
    )
    expanded_selection_note = (
        f"Top {expanded_forget_count} hardest non-public identities by smallest mean cosine distance "
        f"to their top-{hard_retain_count} ArcFace centroid neighbors; requires >= {min_images_per_identity} "
        "images and excludes all public-group identities."
    )

    expanded_groups = []
    for expanded_rank, row in enumerate(hardness.head(expanded_forget_count).itertuples(index=False), start=1):
        forget_id = int(row.target_identity)
        mean_topk_distance = float(row.mean_topk_distance)
        retain_rows = candidate_neighbors.loc[
            (candidate_neighbors["target_identity"] == forget_id)
            & (candidate_neighbors["rank"] <= hard_retain_count)
        ].sort_values("rank")
        retain_ids = [int(x) for x in retain_rows["retain_identity"].tolist()]
        forbidden = {int(forget_id), *retain_ids}
        random_retain_ids = _sample_random_retains(
            rng,
            candidate_identities=eligible["identity"].astype(int).tolist(),
            forbidden=forbidden,
            count=hard_retain_count,
            excluded=public_ids,
        )
        if len(retain_ids) != hard_retain_count or len(random_retain_ids) != hard_retain_count:
            raise ValueError(f"Unable to construct complete non-public retain sets for {forget_id}")
        if public_ids.intersection([forget_id, *retain_ids, *random_retain_ids]):
            raise ValueError(f"Public identity entered expanded group {forget_id}")
        if set(retain_ids).intersection(random_retain_ids):
            raise ValueError(f"Hard/random retain overlap in expanded group {forget_id}")
        all_ids = [forget_id, *retain_ids, *random_retain_ids]
        fit_reference_images = {}
        eval_reference_images = {}
        for identity in all_ids:
            refs = split_reference_images(
                metadata,
                identity,
                fit_count=fit_reference_count,
                eval_count=eval_reference_count,
            )
            fit_reference_images[str(identity)] = refs["fit"]
            eval_reference_images[str(identity)] = refs["eval"]
        expanded_groups.append(
            {
                "set_name": f"Expanded-{forget_id}",
                "forget_id": int(forget_id),
                "retain_ids": retain_ids,
                "hard_retain_ids": retain_ids,
                "random_retain_ids": random_retain_ids,
                "expanded_rank": expanded_rank,
                "expanded_hardness_mean_topk_distance": mean_topk_distance,
                "fit_reference_images": fit_reference_images,
                "eval_reference_images": eval_reference_images,
                "selection_note": (
                    f"{expanded_selection_note} Group rank={expanded_rank}; "
                    f"mean_top{hard_retain_count}_distance={mean_topk_distance:.6f}."
                ),
            }
        )

    global_reference_images, global_eval_reference_images = build_global_reference_maps(
        metadata=metadata,
        identities=eligible["identity"].tolist(),
        fit_reference_count=fit_reference_count,
        eval_reference_count=eval_reference_count,
    )

    return {
        "public_groups": public_groups,
        "expanded_groups": expanded_groups,
        "centroids": centroids.drop(columns=["centroid"]).sort_values("identity").to_dict(orient="records"),
        "neighbors": neighbors.sort_values(["target_identity", "rank"]).to_dict(orient="records"),
        "global_reference_images": global_reference_images,
        "global_eval_reference_images": global_eval_reference_images,
        "reference_split_policy": {
            "fit_reference_count": int(fit_reference_count),
            "eval_reference_count": int(eval_reference_count),
            "fit_rule": "first sorted images by partition,image_name",
            "eval_rule": "next disjoint sorted images by partition,image_name",
        },
        "expanded_selection_policy": {
            "expanded_forget_count": int(expanded_forget_count),
            "hard_retain_count": int(hard_retain_count),
            "min_images_per_identity": int(min_images_per_identity),
            "candidate_exclusions": "public forget and retain identities",
            "hardness_rule": "smallest mean cosine distance to top-k ArcFace centroid neighbors",
            "selection_note": expanded_selection_note,
        },
    }


def dump_benchmark(benchmark: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(benchmark, handle, indent=2)
        handle.write("\n")


def load_benchmark(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def iter_groups(benchmark: dict) -> list[dict]:
    return [*benchmark.get("public_groups", []), *benchmark.get("expanded_groups", [])]


def lookup_group(benchmark: dict, forget_id: int) -> dict:
    for group in iter_groups(benchmark):
        if int(group["forget_id"]) == int(forget_id):
            return group
    raise KeyError(f"No benchmark group for forget identity {forget_id}")


def group_reference_map(group: dict, split: str = "fit") -> dict[str, list[str]]:
    if split == "fit":
        return group.get("fit_reference_images") or group["reference_images"]
    if split == "eval":
        references = group.get("eval_reference_images")
        if not references:
            raise KeyError("Group has no explicit held-out evaluation references")
        return references
    raise ValueError(f"Unsupported reference split: {split}")


def group_reference_list(group: dict, identity: int, split: str = "fit") -> list[str]:
    return list(group_reference_map(group, split=split).get(str(identity), []))


def global_reference_map(benchmark: dict, split: str = "fit") -> dict[str, list[str]]:
    if split == "fit":
        return benchmark.get("global_reference_images", {})
    if split == "eval":
        return benchmark.get("global_eval_reference_images", {})
    raise ValueError(f"Unsupported reference split: {split}")


def global_reference_list(benchmark: dict, identity: int, split: str = "fit") -> list[str]:
    return list(global_reference_map(benchmark, split=split).get(str(identity), []))


def resolve_reference_list(benchmark: dict, group: dict, identity: int, split: str = "fit") -> list[str]:
    refs = group_reference_list(group, identity, split=split)
    if refs:
        return refs
    return global_reference_list(benchmark, identity, split=split)
