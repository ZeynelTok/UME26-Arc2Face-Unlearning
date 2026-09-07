"""Calibrate the AdaFace FAR=0.01 threshold on the train pool."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from genmu_face_unlearning.adaface_utils import AdaFaceEmbedder
from genmu_face_unlearning.face_embedding import landmarks_from_row
from genmu_face_unlearning.threshold import calibrate_arcface_threshold, save_threshold_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "outputs" / "prep" / "real_trainpool_metadata.csv")
    parser.add_argument("--insightface-model-root", type=Path, default=ROOT / "models" / "insightface")
    parser.add_argument("--adaface-weights", type=Path, default=ROOT / "models" / "adaface" / "weights" / "adaface_ir50_webface4m.ckpt")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "prep")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    print(f"Train pool: {len(metadata)} images, {metadata['identity'].nunique()} identities")

    embedder = AdaFaceEmbedder(
        insightface_model_root=args.insightface_model_root,
        adaface_weights=args.adaface_weights,
        device=args.device,
    )

    kept_rows: list[dict] = []
    embeddings: list[np.ndarray] = []
    pending_crops: list[np.ndarray] = []
    pending_rows: list[dict] = []

    def flush() -> None:
        if not pending_crops:
            return
        feats = embedder.embed_aligned_batch(np.stack(pending_crops, axis=0))
        for i, r in enumerate(pending_rows):
            kept_rows.append(r)
            embeddings.append(feats[i])
        pending_crops.clear()
        pending_rows.clear()

    records = metadata.to_dict("records")
    for row in tqdm(records, total=len(records), desc="AdaFace train-pool embeddings", unit="img"):
        image = cv2.imread(row["image_path"])
        if image is None:
            continue
        aligned = embedder._norm_crop(image, landmark=landmarks_from_row(pd.Series(row)))
        pending_crops.append(aligned)
        pending_rows.append(row)
        if len(pending_crops) >= args.batch_size:
            flush()
    flush()

    kept_metadata = pd.DataFrame(kept_rows)
    embedding_matrix = np.stack(embeddings, axis=0)
    print(f"Embedded {len(kept_metadata)} images (dropped {len(metadata) - len(kept_metadata)} unreadable)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "real_trainpool_embeddings_adaface.npy", embedding_matrix)
    kept_metadata.to_csv(args.out_dir / "real_trainpool_metadata_adaface.csv", index=False)

    calibration = calibrate_arcface_threshold(kept_metadata, embedding_matrix, name="adaface_far_0p01")
    save_threshold_calibration(calibration, args.out_dir / "threshold_adaface.json")

    print("AdaFace threshold calibration:")
    for key, value in calibration.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
