"""Compare converted SER-FIQ ONNX output with a no-Dropout MXNet oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--aligned-bgr", type=Path, required=True)
    parser.add_argument("--mxnet-output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--rtol", type=float, default=2e-4)
    parser.add_argument("--atol", type=float, default=2e-4)
    args = parser.parse_args()

    if args.device == "cuda":
        import torch  # noqa: F401  # Preload CUDA/cuDNN DLLs for ORT.

    expected = np.load(args.mxnet_output)
    aligned_bgr = np.load(args.aligned_bgr)
    rgb_chw = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
    batch = np.repeat(rgb_chw[None, :, :, :], expected.shape[0], axis=0).astype(np.float32)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.device == "cuda"
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(args.model), providers=providers)
    mask_shape = tuple(int(value) for value in session.get_inputs()[1].shape)
    mask = np.ones(mask_shape, dtype=np.float32)
    observed = session.run(None, {"data": batch, "dropout_mask": mask})[0]

    difference = np.abs(expected - observed)
    print(f"MXNet shape: {expected.shape}; ONNX shape: {observed.shape}")
    print(f"max_abs_error={difference.max():.9g}; mean_abs_error={difference.mean():.9g}")
    np.testing.assert_allclose(observed, expected, rtol=args.rtol, atol=args.atol)
    print("PASS: converted ONNX operations and parameters match the official MXNet graph.")


if __name__ == "__main__":
    main()
