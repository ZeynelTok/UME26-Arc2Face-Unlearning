from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _default_arc2face_dir() -> Path:
    candidates = [
        ROOT / "models" / "Arc2Face",
        ROOT / "models" / "arc2face",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the core Arc2Face model files from Hugging Face into a local workspace folder.")
    parser.add_argument("--repo-id", type=str, default="FoivosPar/Arc2Face")
    parser.add_argument("--out-dir", type=Path, default=_default_arc2face_dir())
    parser.add_argument(
        "--arcface-recognizer-dir",
        type=Path,
        default=ROOT / "models" / "insightface" / "models" / "antelopev2",
        help="Destination for the official Arc2Face arcface.onnx recognizer.",
    )
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.arcface_recognizer_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "arc2face/config.json",
        "arc2face/diffusion_pytorch_model.safetensors",
        "encoder/config.json",
        "encoder/pytorch_model.bin",
    ]
    for relative_path in files:
        hf_hub_download(
            repo_id=args.repo_id,
            filename=relative_path,
            local_dir=str(args.out_dir),
            local_dir_use_symlinks=False,
        )
        print(f"Downloaded {relative_path} -> {args.out_dir}")

    hf_hub_download(
        repo_id=args.repo_id,
        filename="arcface.onnx",
        local_dir=str(args.arcface_recognizer_dir),
        local_dir_use_symlinks=False,
    )
    print(f"Downloaded arcface.onnx -> {args.arcface_recognizer_dir}")


if __name__ == "__main__":
    main()
