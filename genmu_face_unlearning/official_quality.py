"""SER-FIQ and GraFIQs inference used for the quality evaluation."""
from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


def ser_fiq_scores_from_embeddings(
    embeddings: np.ndarray,
    *,
    alpha: float = 130.0,
    r: float = 0.88,
) -> dict[str, float]:
    """Calculate the paper-native and official normalized SER-FIQ scores.

    This follows ``SER_FIQ.get_score`` in pterhoer/FaceImageQuality commit
    611296605db57b8d50518fd5911d5111eeb52747: L2-normalize each stochastic
    embedding, average all unique pairwise Euclidean distances, apply the
    paper score ``2 / (1 + exp(mean_distance))``, then apply the repository's
    later alpha/r normalization.
    """

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"Expected at least two 2-D embeddings, got {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("SER-FIQ cannot normalize a zero embedding")
    normalized = values / norms
    delta = normalized[:, None, :] - normalized[None, :, :]
    distances = np.sqrt(np.maximum(np.sum(delta * delta, axis=2), 0.0))
    upper = distances[np.triu_indices(values.shape[0], k=1)]
    mean_distance = float(upper.mean())
    paper_score = float(2.0 / (1.0 + math.exp(mean_distance)))
    normalized_score = float(1.0 / (1.0 + math.exp(-alpha * (paper_score - r))))
    return {
        "ser_fiq_mean_euclidean_distance": mean_distance,
        "ser_fiq_paper_score": paper_score,
        "ser_fiq_normalized_score": normalized_score,
    }


def deterministic_image_seed(key: str, *, base_seed: int = 20260810) -> int:
    """Return an order-independent 64-bit seed for one image."""

    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


class OfficialSerFiqOnnxScorer:
    """SER-FIQ (on ArcFace) using the authors' exported MXNet graph.

    The ONNX conversion replaces the graph's always-on Dropout(0.5) node with
    an explicit mask input.  Supplying independently sampled Bernoulli masks
    is mathematically equivalent while making each image reproducible and
    independent of traversal order.
    """

    def __init__(
        self,
        model_path: Path | str,
        *,
        trunk_model_path: Path | str | None = None,
        head_model_path: Path | str | None = None,
        device: str = "cuda",
        base_seed: int = 20260810,
    ) -> None:
        # Importing torch first preloads the CUDA/cuDNN DLLs used by ORT.
        if device == "cuda":
            import torch  # noqa: F401
        import onnxruntime as ort

        self.model_path = Path(model_path)
        self.base_seed = int(base_seed)
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if device == "cuda"
            else ["CPUExecutionProvider"]
        )
        if (trunk_model_path is None) != (head_model_path is None):
            raise ValueError("SER-FIQ trunk and head paths must be supplied together")
        self.trunk_session = None
        self.head_session = None
        if trunk_model_path is not None:
            self.trunk_session = ort.InferenceSession(str(trunk_model_path), providers=providers)
            self.head_session = ort.InferenceSession(str(head_model_path), providers=providers)
            self.session = self.head_session
            trunk_inputs = self.trunk_session.get_inputs()
            if len(trunk_inputs) != 1 or trunk_inputs[0].name != "data":
                raise RuntimeError(f"Unexpected SER-FIQ trunk inputs: {trunk_inputs}")
            if trunk_inputs[0].shape != [1, 3, 112, 112]:
                raise RuntimeError(f"Unexpected SER-FIQ trunk data shape: {trunk_inputs[0].shape}")
        else:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        active = self.session.get_providers()
        if device == "cuda" and (not active or active[0] != "CUDAExecutionProvider"):
            raise RuntimeError(f"SER-FIQ ONNX model did not activate CUDA: {active}")

        inputs = {item.name: item for item in self.session.get_inputs()}
        expected_inputs = {"bn1", "dropout_mask"} if self.trunk_session is not None else {"data", "dropout_mask"}
        if set(inputs) != expected_inputs:
            raise RuntimeError(f"Unexpected SER-FIQ model inputs: {sorted(inputs)}")
        data_name = "bn1" if self.trunk_session is not None else "data"
        data_shape = inputs[data_name].shape
        mask_shape = inputs["dropout_mask"].shape
        expected_data_tail = [512, 7, 7] if self.trunk_session is not None else [3, 112, 112]
        if data_shape[1:] != expected_data_tail:
            raise RuntimeError(f"Unexpected SER-FIQ data shape: {data_shape}")
        if mask_shape[1:] != [512, 7, 7] or mask_shape[0] != data_shape[0]:
            raise RuntimeError(f"Unexpected SER-FIQ mask shape: {mask_shape}")
        self.num_passes = int(data_shape[0])
        if self.num_passes < 2:
            raise RuntimeError("SER-FIQ model must contain at least two stochastic passes")
        self.mask_shape = tuple(int(value) for value in mask_shape)

    def score_aligned_bgr(
        self, aligned_bgr: np.ndarray, *, image_key: str
    ) -> dict[str, float | int | str]:
        image = np.asarray(aligned_bgr)
        if image.shape != (112, 112, 3):
            raise ValueError(f"Expected an aligned 112x112 BGR crop, got {image.shape}")

        # The authors' apply_mtcnn method converts its aligned BGR crop to RGB
        # before passing a CHW array with pixel values in [0, 255] to ArcFace.
        rgb_chw = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
        seed = deterministic_image_seed(image_key, base_seed=self.base_seed)
        rng = np.random.Generator(np.random.PCG64(seed))
        mask = (rng.random(self.mask_shape, dtype=np.float32) >= 0.5).astype(np.float32)
        mask *= 2.0  # inverted Dropout(0.5), matching MXNet's training-mode output

        if self.trunk_session is None:
            repeated = np.repeat(rgb_chw[None, :, :, :], self.num_passes, axis=0)
            model_inputs = {"data": repeated, "dropout_mask": mask}
        else:
            feature = self.trunk_session.run(None, {"data": rgb_chw[None, :, :, :]})[0]
            repeated = np.repeat(feature, self.num_passes, axis=0)
            model_inputs = {"bn1": repeated, "dropout_mask": mask}
        embeddings = self.session.run(None, model_inputs)[0]
        scores = ser_fiq_scores_from_embeddings(embeddings)
        scores["ser_fiq_num_passes"] = self.num_passes
        scores["ser_fiq_seed"] = seed
        scores["ser_fiq_execution"] = (
            "deterministic_trunk_once_then_stochastic_head"
            if self.trunk_session is not None
            else "full_network_repeated"
        )
        return scores


def _load_upstream_grafiqs(source_root: Path):
    script_path = source_root / "extract_grafiqs.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Official GraFIQs source not found: {script_path}")
    source_text = script_path.read_text(encoding="utf-8")
    if "GraFIQs" not in source_text or "get_model" not in source_text:
        raise RuntimeError(f"Unexpected GraFIQs source file: {script_path}")

    root_text = str(source_root)
    sys.path.insert(0, root_text)
    try:
        spec = importlib.util.spec_from_file_location("genmu_official_grafiqs", script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to import {script_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == root_text:
            sys.path.pop(0)


class OfficialGraFIQsScorer:
    """Thin wrapper around jankolf/grafiqs commit 8d11a1f8506f."""

    def __init__(
        self,
        source_root: Path | str,
        weights_path: Path | str,
        *,
        backbone: str = "iresnet100",
        device: str = "cuda",
        bgr2rgb: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        self._device = torch.device("cuda:0" if device == "cuda" else "cpu")
        self._module = _load_upstream_grafiqs(Path(source_root))
        self._model = self._module.get_model(
            nn_architecture=backbone,
            rank=self._device,
            nn_weights_path=str(weights_path),
        )
        self._bgr2rgb = bool(bgr2rgb)
        transforms = self._module.transforms
        self._transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.Resize(
                    size=(112, 112),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def score_aligned_bgr(self, aligned_bgr: np.ndarray) -> dict[str, float]:
        import cv2

        image = np.asarray(aligned_bgr)
        if image.shape != (112, 112, 3):
            raise ValueError(f"Expected an aligned 112x112 BGR crop, got {image.shape}")
        if self._bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = self._transform(image).unsqueeze(0).to(self._device).requires_grad_(True)
        bn_score, (_embedding, block1, block2, block3, block4, _bn) = self._model.get_BN(tensor)
        gradients = self._torch.autograd.grad(
            outputs=bn_score,
            inputs=[tensor, block1, block2, block3, block4],
        )
        names = ["image", "block1", "block2", "block3", "block4"]
        return {
            f"grafiqs_{name}": float(self._torch.abs(gradient[0]).sum().detach().cpu())
            for name, gradient in zip(names, gradients)
        }
