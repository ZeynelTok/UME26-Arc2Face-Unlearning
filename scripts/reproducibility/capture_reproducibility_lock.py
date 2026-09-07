"""Record the software, hardware, and external models used for the paper.

This command works offline and does not change model files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


BASE_REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
BASE_REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def installed_packages() -> list[tuple[str, str]]:
    packages: dict[str, tuple[str, str]] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        packages[name.lower().replace("_", "-")] = (name, dist.version)
    return sorted(packages.values(), key=lambda item: item[0].lower())


def model_files(root: Path) -> list[dict[str, str]]:
    local = [
        ("arc2face_unet", "FoivosPar/Arc2Face", "models/Arc2Face/arc2face/diffusion_pytorch_model.safetensors"),
        ("arc2face_encoder", "FoivosPar/Arc2Face", "models/Arc2Face/encoder/pytorch_model.bin"),
        ("arc2face_model_source", "foivospar/Arc2Face", "models/Arc2Face/arc2face/models.py"),
        ("adaface_ir50_webface4m", "mk-minchul/AdaFace", "models/adaface/weights/adaface_ir50_webface4m.ckpt"),
        ("adaface_model_source", "mk-minchul/AdaFace", "models/adaface/net.py"),
        ("insightface_landmark_3d68", "deepinsight/insightface-antelopev2", "models/insightface/models/antelopev2/1k3d68.onnx"),
        ("insightface_detector_2d106", "deepinsight/insightface-antelopev2", "models/insightface/models/antelopev2/2d106det.onnx"),
        ("insightface_arcface", "deepinsight/insightface-antelopev2", "models/insightface/models/antelopev2/arcface.onnx"),
        ("insightface_genderage", "deepinsight/insightface-antelopev2", "models/insightface/models/antelopev2/genderage.onnx"),
        ("insightface_scrfd_detector", "deepinsight/insightface-antelopev2", "models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx"),
        ("ser_fiq_official_symbol", "pterhoer/FaceImageQuality", "models/official_quality/ser_fiq_source/insightface/model/insightface-symbol.json"),
        ("ser_fiq_official_params", "pterhoer/FaceImageQuality", "models/official_quality/ser_fiq_source/insightface/model/insightface-0000.params"),
        ("ser_fiq_full_onnx", "local validated conversion of pterhoer/FaceImageQuality", "models/official_quality/ser_fiq_arcface_t100_mask.onnx"),
        ("ser_fiq_trunk_onnx", "local validated split of official SER-FIQ", "models/official_quality/ser_fiq_arcface_trunk_b1.onnx"),
        ("ser_fiq_head_onnx", "local validated split of official SER-FIQ", "models/official_quality/ser_fiq_arcface_head_t100.onnx"),
        ("grafiqs_resnet100_ms1mv2_arcface", "jankolf/grafiqs", "models/official_quality/resnet100_ms1mv2_arcface.pth"),
        ("ser_fiq_official_source", "pterhoer/FaceImageQuality", "models/official_quality/ser_fiq_source/face_image_quality.py"),
        ("grafiqs_official_source", "jankolf/grafiqs", "models/official_quality/grafiqs_source/extract_grafiqs.py"),
    ]
    records: list[dict[str, str]] = []
    for name, source, relative in local:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path}")
        print(f"Hashing {relative} ({path.stat().st_size:,} bytes)", flush=True)
        records.append(
            {
                "name": name,
                "source": source,
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    snapshot = (
        cache_root
        / "hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/snapshots"
        / BASE_REVISION
    )
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Pinned Stable Diffusion snapshot is missing: {snapshot}")
    for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
        relative = path.relative_to(snapshot).as_posix()
        print(f"Hashing Stable Diffusion v1.5 {relative} ({path.stat().st_size:,} bytes)", flush=True)
        records.append(
            {
                "name": f"stable_diffusion_v1_5/{relative}",
                "source": BASE_REPO,
                "revision": BASE_REVISION,
                "path": (
                    "hf_cache:hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/"
                    f"snapshots/{BASE_REVISION}/{relative}"
                ),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def capture(root: Path, output_dir: Path) -> None:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    packages = installed_packages()
    lock_lines = [
        "# Exact successful paper environment captured on Windows/Python 3.12.",
        "# PyTorch CUDA wheels are resolved from the official cu128 index.",
        "--extra-index-url https://download.pytorch.org/whl/cu128",
        *[f"{name}=={version}" for name, version in packages],
        "",
    ]
    (output_dir / "requirements-paper-cu128.lock").write_text(
        "\n".join(lock_lines), encoding="utf-8", newline="\n"
    )

    import numpy
    import onnxruntime
    import pandas
    import scipy
    import torch

    cudnn_version = torch.backends.cudnn.version()
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "torch_cuda_available": torch.cuda.is_available(),
        "cudnn_version_integer": cudnn_version,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_available_providers": onnxruntime.get_available_providers(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "nvidia_smi": command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "determinism_note": (
            "Generation uses explicit per-image Torch seeds, but exact cross-GPU/driver "
            "bitwise identity is not promised. Use the released predictions for exact table reproduction."
        ),
        "known_packaging_note": (
            "insightface declares onnxruntime by distribution name; onnxruntime-gpu supplies "
            "the imported module and CUDAExecutionProvider, so pip check may report a false mismatch."
        ),
    }
    write_json(output_dir / "environment.json", environment)

    arc2face_root = root / "models/Arc2Face"
    arc2face_status = command(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=arc2face_root
    )
    models = {
        "stable_diffusion_base": {"repository": BASE_REPO, "revision": BASE_REVISION},
        "arc2face_source": {
            "repository": "https://github.com/foivospar/Arc2Face",
            "git_commit": command(["git", "rev-parse", "HEAD"], cwd=arc2face_root),
            "tracked_worktree_dirty": bool(arc2face_status),
            "note": "The used arc2face/models.py is hashed below; local tracked changes remove comments only.",
        },
        "ser_fiq_source": {
            "repository": "https://github.com/pterhoer/FaceImageQuality",
            "git_commit": command(
                ["git", "rev-parse", "HEAD"], cwd=root / "models/official_quality/ser_fiq_source"
            ),
        },
        "grafiqs_source": {
            "repository": "https://github.com/jankolf/grafiqs",
            "git_commit": command(
                ["git", "rev-parse", "HEAD"], cwd=root / "models/official_quality/grafiqs_source"
            ),
        },
        "files": model_files(root),
    }
    write_json(output_dir / "models.lock.json", models)
    print(f"Captured environment and external-model records in {output_dir.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "paper_artifacts/reproducibility"
    capture(root, output_dir.resolve())


if __name__ == "__main__":
    main()
