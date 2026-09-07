from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_ARCFACE_MODEL_ROOT = ROOT / "models" / "insightface"

from genmu_face_unlearning.arcface_utils import ArcFaceEmbedder
from genmu_face_unlearning.benchmark import load_benchmark, lookup_group, group_reference_map
from genmu_face_unlearning.config import load_experiment_config
from genmu_face_unlearning.embedding_adapter import centroid, select_identity_embedding

FORGET_WEIGHT = 1.0
RETAIN_WEIGHT = 5.0
PRESERVE_WEIGHT = 2.0
REGULARIZATION_WEIGHT = 0.001


def _lazy_import_torch():
    import torch

    return torch


def _reference_embeddings(embedder: ArcFaceEmbedder, reference_paths: list[str]) -> list[np.ndarray]:
    embeddings = []
    for path in reference_paths:
        detection = embedder.detect_and_embed_image(path)
        if detection is None:
            raise RuntimeError(f"Unable to detect face in reference image: {path}")
        embeddings.append(detection.embedding)
    return embeddings


def _benchmark_eval_reference_paths(benchmark: dict) -> set[str]:
    paths = set()
    for refs in benchmark.get("global_eval_reference_images", {}).values():
        paths.update(str(path) for path in refs)
    return paths


def _normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


def _sample_preserve_embeddings(
    metadata_path: Path,
    embeddings_path: Path,
    exclude_ids: set[int],
    exclude_paths: set[str],
    sample_count: int,
    seed: int,
) -> np.ndarray:
    metadata = pd.read_csv(metadata_path, usecols=["identity", "image_path"])
    embeddings = np.load(embeddings_path)
    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embedding row counts differ.")

    keep = ~metadata["identity"].astype(int).isin(exclude_ids)
    if exclude_paths:
        keep &= ~metadata["image_path"].isin(exclude_paths)
    indices = np.flatnonzero(keep.to_numpy())
    rng = np.random.default_rng(seed)
    if len(indices) > sample_count:
        indices = rng.choice(indices, size=sample_count, replace=False)
    return _normalize_matrix(embeddings[indices])


def _cosine_loss(left, right):
    return 1.0 - (left * right).sum(dim=-1)


def _collect_retain_paths(fit_reference_images: dict[str, list[str]], config: dict) -> list[str]:
    retain_paths = []
    for identity in [*config["retain_ids"], *config.get("random_retain_ids", [])]:
        retain_paths.extend(fit_reference_images.get(str(identity), []))
    return retain_paths


def _target_candidate_ids(config: dict, target_identity: int | None) -> list[int]:
    if target_identity is not None:
        return [int(target_identity)]
    return [*config["retain_ids"], *config.get("random_retain_ids", [])]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a low-rank identity-projection adapter for ArcFace embeddings.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arcface-model-root", type=Path, default=DEFAULT_ARCFACE_MODEL_ROOT)
    parser.add_argument(
        "--reference-det-thresh",
        type=float,
        default=0.1,
        help="InsightFace detection threshold for the clean fitting-reference images.",
    )
    parser.add_argument("--metadata", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_metadata.csv")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_embeddings.npy")
    parser.add_argument("--target-identity", type=int, default=None, help="Optional explicit target identity override.")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--preserve-samples", type=int, default=8192)
    parser.add_argument("--preserve-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    torch = _lazy_import_torch()
    torch.manual_seed(args.seed)

    config = load_experiment_config(args.config)
    benchmark = load_benchmark(args.benchmark)
    group = lookup_group(benchmark, config["forget_id"])
    embedder = ArcFaceEmbedder(
        model_root=args.arcface_model_root,
        device=config["generator"]["device"],
        det_thresh=args.reference_det_thresh,
    )
    eval_reference_paths = _benchmark_eval_reference_paths(benchmark)

    fit_reference_images = group_reference_map(group, split="fit")
    forget_paths = list(fit_reference_images[str(config["forget_id"])])
    forget_embeddings = _normalize_matrix(np.stack(_reference_embeddings(embedder, forget_paths), axis=0))
    forget_centroid = centroid(forget_embeddings)

    retain_paths = _collect_retain_paths(fit_reference_images, config)
    retain_embeddings = _normalize_matrix(np.stack(_reference_embeddings(embedder, retain_paths), axis=0))

    candidate_ids = _target_candidate_ids(config, args.target_identity)
    target_identity, target_embedding, target_similarity = select_identity_embedding(
        forget_embedding=forget_centroid,
        forget_id=config["forget_id"],
        benchmark=benchmark,
        metadata_path=args.metadata,
        embeddings_path=args.embeddings,
        mode="nearest",
        candidate_ids=candidate_ids,
    )
    preserve_embeddings = _sample_preserve_embeddings(
        metadata_path=args.metadata,
        embeddings_path=args.embeddings,
        exclude_ids={int(config["forget_id"])},
        exclude_paths=eval_reference_paths,
        sample_count=args.preserve_samples,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and config["generator"]["device"] == "cuda" else "cpu")
    forget_tensor = torch.tensor(forget_embeddings, dtype=torch.float32, device=device)
    retain_tensor = torch.tensor(retain_embeddings, dtype=torch.float32, device=device)
    preserve_tensor = torch.tensor(preserve_embeddings, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_embedding[None, :], dtype=torch.float32, device=device)

    down = torch.nn.Parameter(torch.randn(args.rank, 512, dtype=torch.float32, device=device) * 0.02)
    up = torch.nn.Parameter(torch.zeros(512, args.rank, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW([down, up], lr=args.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)

    def transform(batch):
        residual = (batch @ down.t()) @ up.t()
        return torch.nn.functional.normalize(batch + residual, dim=-1), residual

    progress = tqdm(range(args.steps), desc="train_projection_adapter", unit="step", dynamic_ncols=True)
    last_metrics = {}
    best_score = -float("inf")
    best_metrics = {}
    best_down = None
    best_up = None
    best_step_zero_based = None
    best_optimizer_updates_completed = None
    postfix_every = max(1, args.steps // 100)
    for step in progress:
        batch_indices = rng.integers(0, len(preserve_tensor), size=min(args.preserve_batch_size, len(preserve_tensor)))
        preserve_batch = preserve_tensor[torch.tensor(batch_indices, dtype=torch.long, device=device)]

        forget_out, forget_residual = transform(forget_tensor)
        retain_out, retain_residual = transform(retain_tensor)
        preserve_out, preserve_residual = transform(preserve_batch)

        forget_loss = _cosine_loss(forget_out, target_tensor.expand_as(forget_out)).mean()
        retain_loss = _cosine_loss(retain_out, retain_tensor).mean()
        preserve_loss = _cosine_loss(preserve_out, preserve_batch).mean()
        regularization = (forget_residual.square().mean() + retain_residual.square().mean() + preserve_residual.square().mean())
        loss = (
            FORGET_WEIGHT * forget_loss
            + RETAIN_WEIGHT * retain_loss
            + PRESERVE_WEIGHT * preserve_loss
            + REGULARIZATION_WEIGHT * regularization
        )
        last_metrics = {
            "loss": float(loss.detach().cpu()),
            "forget_cos_to_target": float((forget_out * target_tensor.expand_as(forget_out)).sum(dim=-1).mean().detach().cpu()),
            "retain_cos_to_original": float((retain_out * retain_tensor).sum(dim=-1).mean().detach().cpu()),
            "preserve_cos_to_original": float((preserve_out * preserve_batch).sum(dim=-1).mean().detach().cpu()),
        }
        score = (
            last_metrics["forget_cos_to_target"]
            + 0.5 * last_metrics["retain_cos_to_original"]
            + 0.5 * last_metrics["preserve_cos_to_original"]
        )
        if score > best_score:
            best_score = score
            best_metrics = dict(last_metrics)
            # Save the pre-update parameters used to compute this score.
            best_down = down.detach().cpu().numpy().astype(np.float32)
            best_up = up.detach().cpu().numpy().astype(np.float32)
            best_step_zero_based = int(step)
            best_optimizer_updates_completed = int(step)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or step == args.steps - 1 or (step + 1) % postfix_every == 0:
            progress.set_postfix(last_metrics)

    if best_down is None or best_up is None:
        best_down = down.detach().cpu().numpy().astype(np.float32)
        best_up = up.detach().cpu().numpy().astype(np.float32)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = args.out_dir / "projection_adapter.npz"
    np.savez(
        adapter_path,
        forget_centroid=forget_centroid.astype(np.float32),
        target_embedding=target_embedding.astype(np.float32),
        down=best_down,
        up=best_up,
        rank=np.array(args.rank, dtype=np.int64),
        target_identity=np.array(int(target_identity), dtype=np.int64),
    )
    summary = {
        "adapter_path": str(adapter_path),
        "forget_id": int(config["forget_id"]),
        "forget_reference_paths": forget_paths,
        "target_identity_override": None if args.target_identity is None else int(args.target_identity),
        "target_identity": int(target_identity),
        "target_similarity_to_forget": float(target_similarity),
        "reference_det_thresh": float(args.reference_det_thresh),
        "rank": int(args.rank),
        "steps": int(args.steps),
        "learning_rate": float(args.learning_rate),
        "preserve_samples": int(len(preserve_embeddings)),
        "preserve_batch_size": int(args.preserve_batch_size),
        "seed": int(args.seed),
        "checkpoint_selection_timing": "pre_update_state_scored_and_saved_before_optimizer_step",
        "best_step_zero_based": best_step_zero_based,
        "best_optimizer_updates_completed": best_optimizer_updates_completed,
        "final_metrics": last_metrics,
        "best_metrics": best_metrics,
        "best_score": float(best_score),
    }
    with (args.out_dir / "projection_adapter_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Wrote projection adapter to {adapter_path}")


if __name__ == "__main__":
    main()
