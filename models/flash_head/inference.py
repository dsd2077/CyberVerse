# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml
from loguru import logger

from flash_head.src.distributed.usp_device import get_device, get_parallel_degree
from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "cyberverse_config.yaml"
_FLASH_HEAD_INFER_PARAMS_PATH = "inference.avatar.flash_head.infer_params"
_DEFAULT_RUNTIME_OPTIONS = {
    "compile_model": True,
    "compile_vae": True,
}
_REQUIRED_INFER_PARAM_KEYS = (
    "frame_num",
    "motion_frames_latent_num",
    "tgt_fps",
    "sample_rate",
    "sample_shift",
    "color_correction_strength",
    "cached_audio_duration",
    "num_heads",
    "height",
    "width",
)

_infer_params: dict[str, Any] | None = None
_runtime_options: dict[str, bool] = copy.deepcopy(_DEFAULT_RUNTIME_OPTIONS)


def resolve_config_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    if config_path is None:
        return _DEFAULT_CONFIG_PATH
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _expand_env(raw: str) -> str:
    def _replace_env(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_PATTERN.sub(_replace_env, raw)


def _load_yaml_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = resolve_config_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"FlashHead config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = _expand_env(f.read())

    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"FlashHead config root must be a mapping: {path}")
    return data


def _parse_bool(value: Any, *, key: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"FlashHead config {key} must be a boolean, got {value!r}")


def configure_infer_params(infer_params: dict[str, Any] | None) -> None:
    if not isinstance(infer_params, dict):
        raise ValueError(
            "FlashHead infer_params must be provided at "
            f"{_FLASH_HEAD_INFER_PARAMS_PATH}"
        )

    missing = [key for key in _REQUIRED_INFER_PARAM_KEYS if key not in infer_params]
    if missing:
        raise ValueError(
            "FlashHead infer_params missing required keys: " + ", ".join(missing)
        )

    global _infer_params
    _infer_params = copy.deepcopy(infer_params)


def configure_runtime_options(options: dict[str, Any] | None) -> None:
    if options is not None and not isinstance(options, dict):
        raise ValueError("FlashHead runtime options must be provided as a mapping")

    resolved = copy.deepcopy(_DEFAULT_RUNTIME_OPTIONS)
    source = options or {}
    for key, default in _DEFAULT_RUNTIME_OPTIONS.items():
        resolved[key] = _parse_bool(source.get(key), key=key, default=default)

    global _runtime_options
    _runtime_options = resolved


def load_flash_head_runtime_config(
    config_path: str | os.PathLike[str] | None = None,
    model_name: str = "flash_head",
) -> dict[str, Any]:
    raw = _load_yaml_config(config_path)
    try:
        section = raw["inference"]["avatar"][model_name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"FlashHead config section not found at inference.avatar.{model_name}"
        ) from exc

    if not isinstance(section, dict):
        raise ValueError(f"inference.avatar.{model_name} must be a mapping")

    configure_runtime_options(section)
    configure_infer_params(section.get("infer_params"))
    return copy.deepcopy(section)


def _require_infer_params() -> dict[str, Any]:
    if _infer_params is None:
        raise RuntimeError(
            "FlashHead infer_params are not configured. "
            "Call configure_infer_params(...) or load_flash_head_runtime_config(...) first."
        )
    return _infer_params


def get_runtime_options() -> dict[str, bool]:
    return copy.deepcopy(_runtime_options)


def get_pipeline(world_size, ckpt_dir, model_type, wav2vec_dir):
    infer_params = _require_infer_params()
    runtime_options = get_runtime_options()
    ulysses_degree, ring_degree = get_parallel_degree(world_size, infer_params["num_heads"])
    device = get_device(ulysses_degree, ring_degree)
    logger.info(f"ulysses_degree: {ulysses_degree}, ring_degree: {ring_degree}, device: {device}")

    pipeline = FlashHeadPipeline(
        checkpoint_dir=ckpt_dir,
        model_type=model_type,
        wav2vec_dir=wav2vec_dir,
        device=device,
        use_usp=(world_size > 1),
        compile_model=runtime_options["compile_model"],
        compile_vae=runtime_options["compile_vae"],
    )

    motion_frames_latent_num = infer_params["motion_frames_latent_num"]
    motion_frames_num = (motion_frames_latent_num - 1) * pipeline.config.vae_stride[0] + 1
    infer_params["motion_frames_num"] = motion_frames_num

    if model_type == "pretrained":
        infer_params["sample_steps"] = 20
    else:
        infer_params["sample_steps"] = 4
    return pipeline


def get_base_data(pipeline, cond_image_path_or_dir, base_seed, use_face_crop):
    infer_params = _require_infer_params()
    pipeline.prepare_params(
        cond_image_path_or_dir=cond_image_path_or_dir,
        target_size=(infer_params["height"], infer_params["width"]),
        frame_num=infer_params["frame_num"],
        motion_frames_num=infer_params["motion_frames_num"],
        sampling_steps=infer_params["sample_steps"],
        seed=base_seed,
        shift=infer_params["sample_shift"],
        color_correction_strength=infer_params["color_correction_strength"],
        use_face_crop=use_face_crop,
    )


def get_infer_params():
    return copy.deepcopy(_require_infer_params())


def get_audio_embedding(pipeline, audio_array, audio_start_idx=-1, audio_end_idx=-1):
    infer_params = _require_infer_params()
    # audio_array = loudness_norm(audio_array, infer_params["sample_rate"])
    audio_embedding = pipeline.preprocess_audio(
        audio_array,
        sr=infer_params["sample_rate"],
        fps=infer_params["tgt_fps"],
    )

    if audio_start_idx == -1 or audio_end_idx == -1:
        audio_start_idx = 0
        audio_end_idx = audio_embedding.shape[0]

    indices = (torch.arange(2 * 2 + 1) - 2) * 1

    center_indices = torch.arange(audio_start_idx, audio_end_idx, 1).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=audio_end_idx - 1)

    audio_embedding = audio_embedding[center_indices][None, ...].contiguous()
    return audio_embedding


def run_pipeline(pipeline, audio_embedding):
    audio_embedding = audio_embedding.to(pipeline.device)
    sample = pipeline.generate(audio_embedding)
    sample_frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous()
    return sample_frames
