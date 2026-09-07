from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

ARC2FACE_LOCAL_DIR_NAMES = ("Arc2Face", "arc2face")


def _lazy_import_torch():
    import torch

    return torch


def _lazy_import_diffusers():
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel

    return DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _local_arc2face_candidates() -> list[Path]:
    repo_root = _repo_root()
    return [repo_root / "models" / name for name in ARC2FACE_LOCAL_DIR_NAMES]


def _lazy_import_arc2face():
    for candidate in _local_arc2face_candidates():
        package_init = candidate / "arc2face" / "__init__.py"
        if package_init.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from arc2face import CLIPTextModelWrapper, project_face_embs
    except ImportError as exc:
        raise SystemExit(
            "Arc2Face runtime dependencies are missing. Install the official Arc2Face package or make the official repo importable as `arc2face`."
        ) from exc
    return CLIPTextModelWrapper, project_face_embs


def _resolve_arc2face_source(source: str | Path) -> str:
    candidate = Path(source)
    if candidate.exists():
        return str(candidate)

    candidate_name = candidate.name.lower()
    for alternate in _local_arc2face_candidates():
        if alternate.exists() and alternate.name.lower() == candidate_name:
            return str(alternate)
    return str(source)


def _resolve_dtype(torch, dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_arc2face_components(config: dict, dtype):
    DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel = _lazy_import_diffusers()
    CLIPTextModelWrapper, project_face_embs = _lazy_import_arc2face()
    source = _resolve_arc2face_source(config.get("arc2face_local_dir") or config["arc2face_repo_id"])
    encoder = CLIPTextModelWrapper.from_pretrained(source, subfolder="encoder", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(source, subfolder="arc2face", torch_dtype=dtype)
    pipeline = StableDiffusionPipeline.from_pretrained(
        config["base_model_repo_id"],
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    return project_face_embs, pipeline


def _normalized_embedding_tensor(torch, face_embeddings: np.ndarray, dtype, device):
    tensor = torch.tensor(face_embeddings, dtype=dtype, device=device)
    norm = torch.norm(tensor, dim=1, keepdim=True)
    return torch.where(norm > 0, tensor / norm, tensor)


class Arc2FaceRuntime:
    def __init__(self, config: dict) -> None:
        torch = _lazy_import_torch()
        self.torch = torch
        self._dtype = _resolve_dtype(torch, config["dtype"])
        self.device = config["device"]
        self._project_face_embs, self.pipeline = _load_arc2face_components(config, self._dtype)
        self.pipeline = self.pipeline.to(self.device)

    def project_face_embeddings(self, face_embeddings: np.ndarray):
        tensor = _normalized_embedding_tensor(self.torch, face_embeddings, self._dtype, self.device)
        return self._project_face_embs(self.pipeline, tensor)

    def prompt_embeds_from_face_embedding(self, face_embedding: np.ndarray):
        projected = self.project_face_embeddings(face_embedding[None, :])
        return projected

    def generate_images(
        self,
        face_embedding: np.ndarray,
        seeds: Iterable[int],
        settings: dict,
        num_images_per_seed: int = 1,
        show_progress: bool = True,
    ) -> list[tuple[int, Image.Image]]:
        torch = self.torch
        self.pipeline.set_progress_bar_config(disable=not show_progress)
        prompt_embeds = self.prompt_embeds_from_face_embedding(face_embedding)
        outputs = []
        for seed in seeds:
            generator = torch.Generator(device=self.device).manual_seed(int(seed))
            images = self.pipeline(
                prompt_embeds=prompt_embeds,
                num_inference_steps=settings["num_inference_steps"],
                guidance_scale=settings["guidance_scale"],
                generator=generator,
                num_images_per_prompt=num_images_per_seed,
                height=settings["image_size"],
                width=settings["image_size"],
            ).images
            for image in images:
                outputs.append((int(seed), image))
        self.pipeline.set_progress_bar_config(disable=False)
        return outputs

