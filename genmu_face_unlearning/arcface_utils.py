from __future__ import annotations

import json
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

DEFAULT_ARCFACE_RECOGNIZER = "antelopev2"


def _prepare_onnxruntime_cuda(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        _ = torch.cuda.is_available()
    except Exception:
        pass
    try:
        import onnxruntime as ort

        ort.preload_dlls()
    except Exception:
        pass


def _lazy_import_insightface():
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop

    return FaceAnalysis, norm_crop


def _lazy_import_cv2():
    import cv2

    return cv2


class ArcFaceEmbedder:
    def __init__(
        self,
        model_root: Path | None = None,
        recognizer_name: str = DEFAULT_ARCFACE_RECOGNIZER,
        device: str = "cuda",
        det_thresh: float = 0.5,
        use_fallback: bool = False,
    ) -> None:
        _prepare_onnxruntime_cuda(device)
        FaceAnalysis, norm_crop = _lazy_import_insightface()
        self._norm_crop = norm_crop
        self._model_root = None if model_root is None else Path(model_root)
        self._recognizer_name = recognizer_name
        self._device = device
        self._det_thresh = float(det_thresh)
        self._use_fallback = bool(use_fallback)

        face_analysis_kwargs = {
            "name": recognizer_name,
            "providers": self._providers(),
        }
        if self._model_root is not None:
            face_analysis_kwargs["root"] = str(self._model_root)
        print(
            "Initializing InsightFace FaceAnalysis "
            f"(name={recognizer_name}, device={device}, det_thresh={self._det_thresh}, "
            f"providers={face_analysis_kwargs['providers']})"
        )
        self.app = FaceAnalysis(**face_analysis_kwargs)
        ctx_id = 0 if device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=self._det_thresh)
        if "recognition" not in self.app.models:
            raise RuntimeError(f"InsightFace model pack '{recognizer_name}' did not expose a recognition model.")
        self.recognizer = self.app.models["recognition"]

    def _providers(self) -> list[str]:
        if self._device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        return _normalize_embedding(embedding)

    def embed_aligned_row(self, row: pd.Series) -> DetectionResult:
        cv2 = _lazy_import_cv2()
        image = cv2.imread(row["image_path"])
        if image is None:
            raise FileNotFoundError(f"Unable to read image: {row['image_path']}")
        landmarks = landmarks_from_row(row)
        aligned = self._norm_crop(image, landmark=landmarks)
        embedding = self.normalize_embedding(self.recognizer.get_feat(aligned))
        return DetectionResult(embedding=embedding, landmarks=landmarks, detector_success=True)

    def _fallback_center_crop_embedding(self, image: np.ndarray) -> DetectionResult | None:
        cv2 = _lazy_import_cv2()
        height, width = image.shape[:2]
        if height == 0 or width == 0:
            return None
        side = min(height, width)
        crop_side = max(112, int(side * 0.9))
        crop_side = min(crop_side, side)
        y0 = max((height - crop_side) // 2, 0)
        x0 = max((width - crop_side) // 2, 0)
        crop = image[y0 : y0 + crop_side, x0 : x0 + crop_side]
        if crop.size == 0:
            return None
        aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        embedding = self.normalize_embedding(self.recognizer.get_feat(aligned))
        bbox = np.array([x0, y0, x0 + crop_side, y0 + crop_side], dtype=np.float32)
        return DetectionResult(embedding=embedding, bbox=bbox, detector_success=False)

    def detect_and_embed_image(self, image_path: Path | str) -> DetectionResult | None:
        cv2 = _lazy_import_cv2()
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        return self.detect_and_embed_array(image)

    def detect_and_embed_array(self, image: np.ndarray) -> DetectionResult | None:
        faces = self.app.get(image)
        if not faces:
            if self._use_fallback:
                return self._fallback_center_crop_embedding(image)
            return None
        face = largest_face(faces)
        embedding = self.normalize_embedding(face["embedding"])
        landmarks = np.asarray(face["kps"], dtype=np.float32) if "kps" in face else None
        bbox = np.asarray(face["bbox"], dtype=np.float32)
        return DetectionResult(embedding=embedding, landmarks=landmarks, bbox=bbox, detector_success=True)

    def cosine_similarity(self, left: np.ndarray, right: np.ndarray) -> float:
        return _cosine_similarity(left, right)


def extract_real_image_embeddings(metadata: pd.DataFrame, embedder: ArcFaceEmbedder) -> tuple[pd.DataFrame, np.ndarray]:
    return _extract_real_image_embeddings(metadata, embedder, "ArcFace real-image embeddings")


def save_embedding_artifacts(metadata: pd.DataFrame, embeddings: np.ndarray, out_dir: Path, prefix: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / f"{prefix}_metadata.csv"
    embeddings_path = out_dir / f"{prefix}_embeddings.npy"
    metadata.to_csv(metadata_path, index=False)
    np.save(embeddings_path, embeddings)
    return metadata_path, embeddings_path


def load_embeddings(metadata_path: Path, embeddings_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path)
    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embeddings row counts differ.")
    return metadata, embeddings


def export_detection_summary(summary: dict[str, float | int | str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
