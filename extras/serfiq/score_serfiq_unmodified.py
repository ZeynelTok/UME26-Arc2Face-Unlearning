"""Score one aligned crop with the unmodified authors' SER-FIQ release.

Run inside ``genmu-serfiq-official:mxnet1.8``. This validation script keeps the
upstream random stream, so repeated scores measure the T=100 Monte Carlo noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--aligned-bgr", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--passes", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    aligned_path = args.aligned_bgr.resolve()
    output = args.output.resolve()
    sys.path.insert(0, str(source_root))
    os.chdir(str(source_root))  # Upstream loads model files via relative paths.
    from face_image_quality import SER_FIQ

    aligned_bgr = np.load(aligned_path)
    aligned_rgb_chw = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    scorer = SER_FIQ(gpu=None)
    scores = [float(scorer.get_score(aligned_rgb_chw, T=args.passes)) for _ in range(args.repetitions)]
    payload = {
        "implementation": "unmodified pterhoer/FaceImageQuality face_image_quality.SER_FIQ.get_score",
        "passes": args.passes,
        "repetitions": args.repetitions,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
