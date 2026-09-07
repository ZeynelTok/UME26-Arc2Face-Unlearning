from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULT_EXPERIMENT = {
    "experiment_name": "unnamed_experiment",
    "forget_id": 0,
    "retain_ids": [],
    "hard_retain_ids": [],
    "random_retain_ids": [],
    "eval_seeds": [101, 211, 307, 401, 503, 601, 701, 809],
    "generator": {
        "arc2face_repo_id": "FoivosPar/Arc2Face",
        "base_model_repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "device": "cuda",
        "dtype": "float16",
    },
    "generation": {
        "image_size": 512,
        "num_inference_steps": 25,
        "guidance_scale": 3.0,
    },
}


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def normalize_experiment_config(data: dict) -> dict:
    config = copy.deepcopy(DEFAULT_EXPERIMENT)
    _deep_update(config, data)
    config["forget_id"] = int(config["forget_id"])
    config["retain_ids"] = [int(x) for x in config.get("retain_ids", [])]
    config["hard_retain_ids"] = [int(x) for x in config.get("hard_retain_ids", [])]
    config["random_retain_ids"] = [int(x) for x in config.get("random_retain_ids", [])]
    config["eval_seeds"] = [int(x) for x in config.get("eval_seeds", [])]
    return config


def _strip_default_values(data, default):
    if isinstance(data, dict) and isinstance(default, dict):
        result = {}
        for key, value in data.items():
            if key in default:
                stripped = _strip_default_values(value, default[key])
                if stripped is None:
                    continue
                result[key] = stripped
            elif value not in ({}, [], None):
                result[key] = value
        return result or None
    if isinstance(data, list) and isinstance(default, list):
        return None if data == default else data
    return None if data == default else data


def load_experiment_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return normalize_experiment_config(data)


def save_experiment_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_experiment_config(config)
    minimal = _strip_default_values(normalized, DEFAULT_EXPERIMENT) or {}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(minimal, handle, indent=2)
        handle.write("\n")
