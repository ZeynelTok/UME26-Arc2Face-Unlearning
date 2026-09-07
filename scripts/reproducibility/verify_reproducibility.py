"""Verify the paper environment and required external model files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_locked_path(root: Path, value: str) -> Path:
    if value.startswith("hf_cache:"):
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        return cache_root / value.removeprefix("hf_cache:")
    return root / value


def verify(root: Path, lock_dir: Path, verify_model_hashes: bool) -> None:
    failures: list[str] = []
    lock_path = lock_dir / "requirements-paper-cu128.lock"
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, expected = line.split("==", 1)
        try:
            observed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"missing package: {name}=={expected}")
            continue
        if observed != expected:
            failures.append(f"package {name}: expected {expected}, observed {observed}")

    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        failures.append("PyTorch cannot access CUDA")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        failures.append("ONNX Runtime does not expose CUDAExecutionProvider")

    models = json.loads((lock_dir / "models.lock.json").read_text(encoding="utf-8"))
    for record in models["files"]:
        path = resolve_locked_path(root, record["path"])
        if not path.is_file():
            failures.append(f"missing external model/source file: {record['path']}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"model/source size mismatch: {record['path']}")
            continue
        if verify_model_hashes:
            print(f"Verifying {record['name']}", flush=True)
            if sha256_file(path) != record["sha256"]:
                failures.append(f"external model/source hash mismatch: {record['path']}")

    if failures:
        raise SystemExit("Reproducibility verification failed:\n- " + "\n- ".join(failures))
    print("Environment and external-model checks passed")
    print(f"  packages: {sum(1 for line in lock_path.read_text().splitlines() if line and not line.startswith(('#', '--')))}")
    print(
        f"  external model/source files: {len(models['files'])} "
        f"({'hashed' if verify_model_hashes else 'size checked'})"
    )
    print(f"  torch CUDA: {torch.__version__}; ONNX providers: {ort.get_available_providers()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--lock-dir", type=Path)
    parser.add_argument("--verify-model-hashes", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock_dir = (args.lock_dir or root / "paper_artifacts/reproducibility").resolve()
    verify(root, lock_dir, args.verify_model_hashes)


if __name__ == "__main__":
    main()
