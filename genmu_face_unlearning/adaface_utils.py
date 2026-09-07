"""AdaFace IR-50 wrapper used for the second-recognizer check."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .face_embedding import (
    DetectionResult,
    cosine_similarity as _cosine_similarity,
    extract_real_image_embeddings as _extract_real_image_embeddings,
    landmarks_from_row,
    largest_face,
    normalize_embedding as _normalize_embedding,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAFACE_ROOT = ROOT / "models" / "adaface"
DEFAULT_ADAFACE_WEIGHTS = DEFAULT_ADAFACE_ROOT / "weights" / "adaface_ir50_webface4m.ckpt"
DEFAULT_ADAFACE_ARCH = "ir_50"


def _load_adaface_net_module(adaface_root: Path):
    net_path = adaface_root / "net.py"
    if not net_path.exists():
        raise FileNotFoundError(
            f"AdaFace net.py not found at {net_path}. Expected the official AdaFace "
            "architecture file (models/adaface/net.py)."
        )
    spec = importlib.util.spec_from_file_location("adaface_net", net_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["adaface_net"] = module
    spec.loader.exec_module(module)
    return module


class AdaFaceEmbedder:
    def __init__(
        self,
        insightface_model_root: Path | None = None,
        adaface_root: Path | None = None,
        adaface_weights: Path | None = None,
        adaface_arch: str = DEFAULT_ADAFACE_ARCH,
        recognizer_name: str = "antelopev2",
        device: str = "cuda",
        det_thresh: float = 0.5,
    ) -> None:
        import torch

        from insightface.app import FaceAnalysis
        from insightface.utils.face_align import norm_crop

        self._torch = torch
        self._norm_crop = norm_crop
        self._device = device
        self._det_thresh = float(det_thresh)
        self._insightface_model_root = None if insightface_model_root is None else Path(insightface_model_root)

        face_analysis_kwargs = {
            "name": recognizer_name,
            "providers": (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if device == "cuda"
                else ["CPUExecutionProvider"]
            ),
        }
        if self._insightface_model_root is not None:
            face_analysis_kwargs["root"] = str(self._insightface_model_root)
        print(
            "Initializing InsightFace FaceAnalysis for AdaFace detection/alignment "
            f"(name={recognizer_name}, device={device}, det_thresh={self._det_thresh})"
        )
        self.app = FaceAnalysis(**face_analysis_kwargs)
        ctx_id = 0 if device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=self._det_thresh)

        adaface_root = Path(adaface_root) if adaface_root is not None else DEFAULT_ADAFACE_ROOT
        adaface_weights = Path(adaface_weights) if adaface_weights is not None else DEFAULT_ADAFACE_WEIGHTS
        net = _load_adaface_net_module(adaface_root)
        print(f"Loading AdaFace {adaface_arch} from {adaface_weights}")
        model = net.build_model(adaface_arch)
        state_dict = torch.load(adaface_weights, map_location="cpu", weights_only=False)["state_dict"]
        # Official AdaFace convention: strip the 'model.' prefix, drop the training head.
        model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"AdaFace state_dict load mismatch. missing={missing} unexpected={unexpected}"
            )
        model.eval()
        self._model = model.to(device)

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        return _normalize_embedding(embedding)

    def _embed_aligned_crop(self, aligned_bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        arr = aligned_bgr.astype(np.float32)
        arr = ((arr / 255.0) - 0.5) / 0.5
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feature, _norm = self._model(tensor)
        embedding = feature.squeeze(0).detach().cpu().numpy()
        return self.normalize_embedding(embedding)

    def embed_aligned_batch(self, aligned_bgr_batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        batch = aligned_bgr_batch.astype(np.float32)
        batch = ((batch / 255.0) - 0.5) / 0.5
        tensor = torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2))).to(self._device)
        with torch.no_grad():
            features, _norm = self._model(tensor)
        features = features.detach().cpu().numpy()
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (features / norms).astype(np.float32)

    def embed_aligned_row(self, row: pd.Series) -> DetectionResult:
        import cv2

        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {row['image_path']}")
        landmarks = landmarks_from_row(row)
        aligned = self._norm_crop(image, landmark=landmarks)
        embedding = self._embed_aligned_crop(aligned)
        return DetectionResult(embedding=embedding, landmarks=landmarks, detector_success=True)

    def detect_and_embed_image(self, image_path: Path | str) -> DetectionResult | None:
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return self.detect_and_embed_array(image)

    def detect_and_embed_array(self, image: np.ndarray) -> DetectionResult | None:
        faces = self.app.get(image)
        if not faces:
            return None
        face = largest_face(faces)
        landmarks = np.asarray(face["kps"], dtype=np.float32)
        aligned = self._norm_crop(image, landmark=landmarks)
        embedding = self._embed_aligned_crop(aligned)
        bbox = np.asarray(face["bbox"], dtype=np.float32)
        return DetectionResult(embedding=embedding, landmarks=landmarks, bbox=bbox, detector_success=True)

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return _cosine_similarity(left, right)


def extract_real_image_embeddings(metadata: pd.DataFrame, embedder: AdaFaceEmbedder) -> tuple[pd.DataFrame, np.ndarray]:
    return _extract_real_image_embeddings(metadata, embedder, "AdaFace real-image embeddings")
