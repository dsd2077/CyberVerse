import asyncio
import logging
import os
import threading
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Iterator

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from inference.core.types import AudioChunk, PluginConfig, VideoChunk
from inference.plugins.avatar.base import AvatarPlugin
from inference.plugins.avatar.warmup import resolve_avatar_warmup_policy

logger = logging.getLogger(__name__)

_sys_path_lock = threading.Lock()
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_DIST_OP_INFER = 0
_DIST_OP_SHUTDOWN = 1
_DIST_OP_RESET = 2
_DIST_OP_KEEPALIVE = 3
_DIST_OP_SET_AVATAR = 4


def _get_infer_param(config: PluginConfig, key: str, default: object) -> object:
    infer_params = config.params.get("infer_params")
    if isinstance(infer_params, dict) and key in infer_params:
        return infer_params[key]
    return config.params.get(key, default)


def _audio_bytes_to_float32_mono(data: bytes, format_hint: str) -> np.ndarray:
    """Decode raw audio bytes to mono float32 in [-1, 1]."""
    fmt = (format_hint or "").strip().lower()
    if fmt in ("float32", "f32", "pcm_f32le"):
        b = data
        if len(b) % 4:
            b = b[: len(b) - (len(b) % 4)]
        if not b:
            return np.array([], dtype=np.float32)
        return np.frombuffer(b, dtype="<f4").copy()
    b = data
    if len(b) % 2:
        b = b[: len(b) - 1]
    if not b:
        return np.array([], dtype=np.float32)
    return (np.frombuffer(b, dtype="<i2").astype(np.float32) / 32768.0).copy()


def _resample_linear_mono(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if x.size == 0 or src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n_src = int(x.shape[0])
    n_dst = max(int(round(n_src * dst_sr / src_sr)), 1)
    t_src = np.arange(n_src, dtype=np.float64) / float(src_sr)
    t_end = (n_src - 1) / float(src_sr) if n_src > 1 else 0.0
    t_dst = np.linspace(0.0, t_end, n_dst, dtype=np.float64)
    return np.interp(t_dst, t_src, x.astype(np.float64)).astype(np.float32)


def _ensure_distributed_env(world_size: int) -> None:
    if world_size <= 1:
        return
    required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "LiveAct world_size>1 requires distributed launch env vars. "
            f"Missing: {', '.join(missing)}. "
            "Use torchrun and set world_size to match WORLD_SIZE."
        )
    env_ws = int(os.environ["WORLD_SIZE"])
    if env_ws != int(world_size):
        raise RuntimeError(
            f"LiveAct world_size mismatch: config={world_size}, WORLD_SIZE={env_ws}."
        )


def _apply_cuda_visible_devices(config: PluginConfig) -> None:
    raw = config.params.get("cuda_visible_devices")
    if raw is None:
        return
    value = str(raw).strip()
    if not value:
        raise ValueError("cuda_visible_devices is set but empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("LiveAct using CUDA_VISIBLE_DEVICES=%s", value)


def _parse_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    normalized = str(value).strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_positive_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"Expected a positive float, got {value!r}")
    return parsed


def _is_primary_rank(rank: int, world_size: int) -> bool:
    return world_size <= 1 or rank == 0


def _dist_barrier(device: int | None = None) -> None:
    if not dist.is_initialized():
        return

    backend = dist.get_backend()
    backend_name = str(backend).lower()
    if device is not None and (
        backend == dist.Backend.NCCL or backend_name.endswith("nccl")
    ):
        dist.barrier(device_ids=[device])
        return
    dist.barrier()


def _distributed_all_ranks_ready(local_ready: bool, device: int | None = None) -> bool:
    """Synchronize avatar init readiness across ranks before distributed warmup."""
    if not dist.is_initialized():
        return local_ready

    tensor_device = (
        torch.device(f"cuda:{device}")
        if device is not None and torch.cuda.is_available()
        else "cpu"
    )
    ready = torch.tensor(
        [1 if local_ready else 0],
        dtype=torch.int32,
        device=tensor_device,
    )
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    return bool(int(ready.item()))


class LiveActAvatarPlugin(AvatarPlugin):
    """Wraps SoulX-LiveAct inference as an Avatar plugin.

    Key design:
    - 18B diffusion model (Wan2.1 + audio module) with 4-step denoising
    - Maintains KV cache (3 timesteps x 40 layers) across iterations for temporal consistency
    - Audio accumulation buffer: triggers generation when enough audio for one iteration
    - VAE overlap decoding for smooth frame transitions
    - Thread lock for GPU serialization
    """

    name = "avatar.live_act"

    # ── Constants ──────────────────────────────────────────────────────────
    VAE_STRIDE = (4, 8, 8)
    PATCH_SIZE = (1, 2, 2)
    BLKSZ_LST = [6, 8]
    TIMESTEP_VALUES = [1000.0, 937.5, 833.33333333, 0.0]
    NUM_LAYERS = 40
    HEAD_DIM = 128
    NUM_HEADS = 40

    def __init__(self) -> None:
        # Models (set during _init_sync)
        self._wan_model = None
        self._vae = None
        self._clip = None
        self._text_encoder = None
        self._audio_encoder = None
        self._wav2vec_fe = None

        # Cached function references from util_liveact
        self._fn_get_audio_emb = None
        self._fn_get_embedding = None
        self._fn_get_msk = None
        self._fn_center_crop = None

        # KV cache
        self._kv_cache: dict | None = None
        self._timesteps: list | None = None

        # Streaming state
        self._pre_latent: torch.Tensor | None = None
        self._iteration_count: int = 0
        self._raw_audio: np.ndarray = np.array([], dtype=np.float32)
        self._raw_audio_start_sample: int = 0
        self._chunk_counter: int = 0

        # Cached encodings for current avatar
        self._clip_context: torch.Tensor | None = None
        self._y: torch.Tensor | None = None
        self._msk: torch.Tensor | None = None
        self._context: list | None = None
        self._ref_target_masks: torch.Tensor | None = None
        self._transform = None

        # Config
        self._fps: int = 24
        self._height: int = 832
        self._width: int = 480
        self._seed: int = 42
        self._audio_cfg: float = 1.0
        self._device: int = 0
        self._t5_cpu: bool = True
        self._fp8_kv_cache: bool = False
        self._offload_cache: bool = False
        self._block_offload: bool = False
        self._mean_memory: bool = False
        self._default_prompt: str = "一个人在说话"

        # Derived constants (set in _init_sync)
        self._frame_num: int = 0
        self._frame_len: int = 0
        self._kv_cache_tokens: int = 0

        # Concurrency
        self._lock = threading.Lock()
        self._avatar_initialized: bool = False
        self._default_avatar_path: str | None = None
        self._default_avatar_is_temp: bool = False

        # Distributed
        self._rank: int = int(os.environ.get("RANK", "0"))
        self._world_size: int = 1
        self._dist_control_lock = threading.Lock()
        self._dist_worker_thread: threading.Thread | None = None
        self._dist_worker_stop = threading.Event()
        self._dist_keepalive_thread: threading.Thread | None = None
        self._dist_keepalive_stop = threading.Event()
        self._dist_keepalive_interval_s: float = 30.0
        self._dist_keepalive_idle_s: float = 30.0
        self._dist_last_command_monotonic: float = 0.0

    # ── Plugin lifecycle ──────────────────────────────────────────────────

    async def initialize(self, config: PluginConfig) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_sync, config)

    def _init_sync(self, config: PluginConfig) -> None:
        _apply_cuda_visible_devices(config)

        world_size = int(config.params.get("world_size", 1))
        if world_size < 1:
            raise ValueError(f"Invalid world_size={world_size}")
        self._world_size = world_size
        _ensure_distributed_env(world_size)

        self._rank = int(os.environ.get("RANK", "0"))
        self._device = int(os.environ.get("LOCAL_RANK", "0"))
        self._dist_keepalive_interval_s = _parse_positive_float(
            os.environ.get(
                "LIVEACT_DIST_KEEPALIVE_INTERVAL_S",
                config.params.get("dist_keepalive_interval_s"),
            ),
            default=30.0,
        )
        self._dist_keepalive_idle_s = _parse_positive_float(
            os.environ.get(
                "LIVEACT_DIST_KEEPALIVE_IDLE_S",
                config.params.get("dist_keepalive_idle_s"),
            ),
            default=self._dist_keepalive_interval_s,
        )
        if self._dist_keepalive_idle_s < self._dist_keepalive_interval_s:
            self._dist_keepalive_idle_s = self._dist_keepalive_interval_s
        warmup_policy = resolve_avatar_warmup_policy(
            config,
            world_size=self._world_size,
        )

        # Parse config
        size_str = _get_infer_param(config, "size", "480*832")
        self._width, self._height = [int(x) for x in size_str.split("*")]
        self._fps = int(_get_infer_param(config, "fps", 24))
        self._seed = int(config.params.get("seed", 42))
        self._audio_cfg = float(_get_infer_param(config, "audio_cfg", 1.0))
        self._t5_cpu = bool(config.params.get("t5_cpu", True))
        self._fp8_kv_cache = bool(config.params.get("fp8_kv_cache", False))
        self._offload_cache = bool(config.params.get("offload_cache", False))
        self._block_offload = bool(config.params.get("block_offload", False))
        self._mean_memory = bool(config.params.get("mean_memory", False))
        self._default_prompt = config.params.get("default_prompt", "一个人在说话")

        # Derived constants
        self._frame_num = (sum(self.BLKSZ_LST) - 1) * self.VAE_STRIDE[0] + 1  # 53
        self._frame_len = (
            (self._height // (self.PATCH_SIZE[1] * self.VAE_STRIDE[1]))
            * (self._width // (self.PATCH_SIZE[2] * self.VAE_STRIDE[2]))
        )
        self._kv_cache_tokens = self._frame_len * sum(self.BLKSZ_LST) // self._world_size

        # Add LiveAct source to sys.path
        models_dir = config.params.get("models_dir")
        if models_dir:
            import sys
            resolved = str(Path(models_dir).resolve())
            with _sys_path_lock:
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)

        # Init distributed
        if self._world_size > 1:
            if not dist.is_initialized():
                torch.cuda.set_device(self._device)
                init_kwargs = {
                    "backend": "nccl",
                    "init_method": "env://",
                    "rank": self._rank,
                    "world_size": self._world_size,
                }
                try:
                    dist.init_process_group(
                        device_id=torch.device(f"cuda:{self._device}"),
                        **init_kwargs,
                    )
                except TypeError:
                    dist.init_process_group(**init_kwargs)
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(rank=self._rank, world_size=self._world_size)
            initialize_model_parallel(
                sequence_parallel_degree=self._world_size,
                ring_degree=1,
                ulysses_degree=self._world_size,
            )

        self._load_models(config)
        self._init_kv_cache()

        # Use a gray placeholder avatar for initialization and warmup.
        base_seed = self._seed
        try:
            image_path, is_temp = self._create_default_avatar_placeholder()
            self._default_avatar_path = image_path
            self._default_avatar_is_temp = is_temp
            self._set_avatar_sync_local(image_path)
            avatar_ready = _distributed_all_ranks_ready(
                self._avatar_initialized,
                self._device,
            )
            if warmup_policy.enabled and avatar_ready:
                self._warmup()
            elif _is_primary_rank(self._rank, self._world_size):
                logger.info(
                    "LiveAct warmup skipped: avatar_ready=%s global_enabled=%s distributed_enabled=%s world_size=%d",
                    avatar_ready,
                    warmup_policy.global_enabled,
                    warmup_policy.distributed_enabled,
                    self._world_size,
                )
            if _is_primary_rank(self._rank, self._world_size):
                logger.info(
                    "LiveAct initialized: size=%dx%d fps=%d ckpt=%s wav2vec=%s "
                    "avatar=%s seed=%d world_size=%d device=%d",
                    self._width, self._height, self._fps,
                    config.params.get("ckpt_dir", ""),
                    config.params.get("wav2vec_dir", ""),
                    image_path, base_seed, self._world_size, self._device,
                )
        except Exception:
            logger.exception("LiveAct default avatar init failed")
            self._avatar_initialized = False

        # Distributed worker for non-rank-0
        dist_worker_main = _parse_bool(
            os.environ.get(
                "LIVEACT_DIST_WORKER_MAIN_THREAD",
                config.params.get("dist_worker_main_thread"),
            ),
            default=False,
        )
        if dist_worker_main and self._world_size > 1 and self._rank != 0:
            self._dist_worker_loop()
        elif not dist_worker_main:
            self._start_dist_worker_if_needed()

        self._start_dist_keepalive_if_needed()

    def _load_models(self, config: PluginConfig) -> None:
        import torchaudio  # noqa: F401 - ensure available
        from torchvision import transforms

        ckpt_dir = config.params["ckpt_dir"]
        wav2vec_dir = config.params["wav2vec_dir"]
        device = self._device
        compile_wan_model = _parse_bool(
            os.environ.get(
                "LIVEACT_COMPILE_WAN_MODEL",
                config.params.get("compile_wan_model"),
            ),
            default=False,
        )
        compile_vae_decode = _parse_bool(
            os.environ.get(
                "LIVEACT_COMPILE_VAE_DECODE",
                config.params.get("compile_vae_decode"),
            ),
            default=False,
        )
        if _is_primary_rank(self._rank, self._world_size):
            logger.info(
                "LiveAct torch.compile: wan_model=%s vae_decode=%s",
                compile_wan_model,
                compile_vae_decode,
            )

        # Import LiveAct modules
        from util_liveact import (
            center_rescale_crop_keep_ratio,
            get_audio_emb,
            get_embedding,
            get_msk,
        )
        self._fn_center_crop = center_rescale_crop_keep_ratio
        self._fn_get_audio_emb = get_audio_emb
        self._fn_get_embedding = get_embedding
        self._fn_get_msk = get_msk

        # WAN model
        if self._world_size > 1:
            from model_liveact.model_memory_sp import WanModel
        else:
            from model_liveact.model_memory import WanModel

        self._wan_model = WanModel.from_pretrained(
            ckpt_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False
        ).to(dtype=torch.bfloat16)

        from fp8_gemm import FP8GemmOptions, enable_fp8_gemm
        enable_fp8_gemm(self._wan_model, options=FP8GemmOptions())

        if self._block_offload:
            for name, child in self._wan_model.named_children():
                if name != "blocks":
                    child.to(device)
            self._wan_model.enable_block_offload(
                onload_device=torch.device(f"cuda:{device}"),
            )
        else:
            self._wan_model = self._wan_model.to(device)

        self._wan_model.freqs = self._wan_model.freqs.to(device)
        self._wan_model.eval()
        if compile_wan_model:
            self._wan_model = torch.compile(
                self._wan_model, mode="max-autotune-no-cudagraphs",
                backend="inductor", dynamic=True,
            )

        # Init kv indices for each block
        for n in range(self.NUM_LAYERS):
            self._wan_model.blocks[n].self_attn.init_kvidx(
                self._frame_len, self._world_size
            )

        # VAE
        from lightx2v.models.video_encoders.hf.wan.vae import WanVAE as LightVAE
        self._vae = LightVAE(
            vae_path=os.path.join(ckpt_dir, "Wan2.1_VAE.pth"),
            dtype=torch.bfloat16, device=device,
            use_lightvae=False, parallel=(self._world_size > 1),
        )
        self._vae.model.eval()
        if compile_vae_decode:
            decode_attr = "tiled_decode" if self._vae.use_tiling else "decode"
            decode_fn = getattr(self._vae.model, decode_attr)
            setattr(
                self._vae.model,
                decode_attr,
                torch.compile(
                    decode_fn,
                    mode="max-autotune-no-cudagraphs",
                    backend="inductor",
                    dynamic=True,
                ),
            )

        # CLIP
        from wan.modules.clip import CLIPModel
        self._clip = CLIPModel(
            checkpoint_path=os.path.join(
                ckpt_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
            ),
            tokenizer_path=os.path.join(ckpt_dir, "xlm-roberta-large"),
            dtype=torch.bfloat16, device=device,
        )

        # T5
        from wan.modules.t5 import T5EncoderModel
        t5_device = "cpu" if self._t5_cpu else device
        self._text_encoder = T5EncoderModel(
            text_len=512, dtype=torch.bfloat16, device=t5_device,
            checkpoint_path=os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(ckpt_dir, "google/umt5-xxl"),
        )

        # Audio encoder
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model
        from transformers import Wav2Vec2FeatureExtractor
        self._audio_encoder = (
            Wav2Vec2Model.from_pretrained(
                wav2vec_dir, local_files_only=True, torch_dtype=torch.bfloat16
            )
            .to(device, dtype=torch.bfloat16)
            .eval()
        )
        self._wav2vec_fe = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )
        self._audio_encoder.feature_extractor._freeze_parameters()

        # Freeze all
        for model in [self._wan_model, self._clip.model, self._audio_encoder, self._vae.model]:
            for param in model.parameters():
                param.requires_grad = False

        # Image transform
        height, width = self._height, self._width
        self._transform = transforms.Compose([
            transforms.Lambda(
                lambda img: self._fn_center_crop(img, (height, width))
            ),
            transforms.ToTensor(),
            transforms.Resize((height, width)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Timestep tensors
        self._timesteps = [
            torch.tensor([v]).to(device, dtype=torch.float32)
            for v in self.TIMESTEP_VALUES
        ]

        torch.cuda.empty_cache()

    def _init_kv_cache(self) -> None:
        kv_device = "cpu" if self._offload_cache else self._device
        kv_dtype = torch.float8_e4m3fn if self._fp8_kv_cache else torch.bfloat16
        kv_scale_shape = (1, self._kv_cache_tokens, self.NUM_HEADS, 1)
        n_steps = len(self.TIMESTEP_VALUES) - 1  # 3

        self._kv_cache = {
            i: {
                layer_id: {
                    "k": torch.zeros(
                        [1, self._kv_cache_tokens, self.NUM_HEADS, self.HEAD_DIM],
                        dtype=kv_dtype, device=kv_device,
                    ),
                    "v": torch.zeros(
                        [1, self._kv_cache_tokens, self.NUM_HEADS, self.HEAD_DIM],
                        dtype=kv_dtype, device=kv_device,
                    ),
                    "k_scale": (
                        torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_device)
                        if self._fp8_kv_cache else None
                    ),
                    "v_scale": (
                        torch.ones(kv_scale_shape, dtype=torch.float32, device=kv_device)
                        if self._fp8_kv_cache else None
                    ),
                    "mean_memory": self._mean_memory,
                    "offload_cache": self._offload_cache,
                    "fp8_kv_cache": self._fp8_kv_cache,
                }
                for layer_id in range(self.NUM_LAYERS)
            }
            for i in range(n_steps)
        }

    def _zero_kv_cache(self) -> None:
        if self._kv_cache is None:
            return
        for step_cache in self._kv_cache.values():
            for layer_cache in step_cache.values():
                layer_cache["k"].zero_()
                layer_cache["v"].zero_()
                if layer_cache.get("k_scale") is not None:
                    layer_cache["k_scale"].fill_(1.0)
                if layer_cache.get("v_scale") is not None:
                    layer_cache["v_scale"].fill_(1.0)

    def _reset_streaming_state(self) -> None:
        self._zero_kv_cache()
        self._pre_latent = None
        self._iteration_count = 0
        self._raw_audio = np.array([], dtype=np.float32)
        self._raw_audio_start_sample = 0
        self._chunk_counter = 0
        torch.manual_seed(self._seed)

    def _warmup(self) -> None:
        if (
            not self._avatar_initialized
            or self._clip_context is None
            or self._y is None
            or self._context is None
            or self._ref_target_masks is None
        ):
            logger.info("[Warmup][Rank %d] skipped: avatar context not initialized", self._rank)
            return

        logger.info("[Warmup][Rank %d] start", self._rank)
        _dist_barrier(self._device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self._device)

        try:
            with torch.no_grad():
                frame_num = self._frame_num
                device = self._device

                # Dummy audio
                dummy_audio = torch.randn(16000 * 6)
                audio_embedding = self._fn_get_embedding(
                    dummy_audio, self._wav2vec_fe, self._audio_encoder, device=device,
                )
                clip_context = self._clip_context
                ref_target_masks = self._ref_target_masks
                y = self._y
                context = self._context
                y_cut = y[:, :, :frame_num // 4 + 1, ...]

                # 3 warmup iterations:
                # 1) first chunk path
                # 2) overlap decode path
                # 3) steady-state update_cache=True path
                pre_latent = None
                total_iterations = 3
                for iteration in range(total_iterations):
                    audio_start_idx = 0 if iteration == 0 else (iteration - 1) * self.BLKSZ_LST[-1] * self.VAE_STRIDE[0]
                    audio_end_idx = audio_start_idx + frame_num
                    audio_embs = self._fn_get_audio_emb(
                        audio_embedding, audio_start_idx, audio_end_idx, device,
                    )
                    f_idx = 0 if iteration == 0 else 1
                    latent = torch.randn(
                        16, self.BLKSZ_LST[f_idx],
                        self._height // self.VAE_STRIDE[1],
                        self._width // self.VAE_STRIDE[2],
                        dtype=torch.bfloat16, device=device,
                    )

                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        for i in range(len(self._timesteps) - 1):
                            arg_c = {
                                "context": context,
                                "clip_fea": clip_context,
                                "ref_target_masks": ref_target_masks,
                                "audio": audio_embs,
                                "y": y_cut[:, :, sum(self.BLKSZ_LST[:f_idx]):sum(self.BLKSZ_LST[:f_idx + 1])],
                                "start_idx": sum(self.BLKSZ_LST[:f_idx]) * self._frame_len,
                                "end_idx": sum(self.BLKSZ_LST[:f_idx + 1]) * self._frame_len,
                                "update_cache": iteration > 1,
                            }
                            noise_pred = self._wan_model(
                                [latent], t=self._timesteps[i],
                                kv_cache=self._kv_cache[i],
                                skip_audio=i not in (1, 2),
                                **arg_c,
                            )[0]
                            dt = (self._timesteps[i] - self._timesteps[i + 1]) / 1000
                            latent = latent + (-noise_pred) * dt[0]

                        if iteration == 0:
                            _videos = self._vae.decode(latent)
                        else:
                            combined = torch.cat([pre_latent[:, -3:], latent], dim=1)
                            _videos = self._vae.decode(combined)[:, :, 9:]
                        pre_latent = latent

                    torch.cuda.synchronize(device)
                    logger.info(
                        "[Warmup][Rank %d] iteration %d/%d done",
                        self._rank,
                        iteration + 1,
                        total_iterations,
                    )

            self._reset_streaming_state()
            torch.cuda.synchronize(device)

            _dist_barrier(self._device)
            logger.info("[Warmup][Rank %d] done", self._rank)

        except Exception as e:
            logger.exception("[Warmup][Rank %d] failed: %s", self._rank, e)
            raise

    # ── Avatar setup ──────────────────────────────────────────────────────

    def _create_default_avatar_placeholder(self) -> tuple[str, bool]:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        img = Image.new("RGB", (self._width, self._height), color=(128, 128, 128))
        img.save(tmp_path, format="PNG")
        return tmp_path, True

    async def set_avatar(self, image_path: str, use_face_crop: bool = False) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._set_avatar_sync, image_path)

    def _set_avatar_sync(self, image_path: str) -> None:
        with self._lock:
            if self._world_size > 1 and self._rank == 0:
                self._distributed_set_avatar(image_path)
            else:
                self._set_avatar_sync_local(image_path)

    def _set_avatar_sync_local(self, image_path: str) -> None:
        device = self._device

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        cond_image = (
            self._transform(image)
            .unsqueeze(1).unsqueeze(0)
            .to(device, torch.bfloat16)
        )  # [1, 3, 1, H, W]

        # CLIP encode
        self._clip.model.to(device)
        with torch.no_grad():
            self._clip_context = self._clip.visual(cond_image)  # [1, 257, 1280]
        self._clip.model.cpu()
        torch.cuda.empty_cache()

        # VAE encode reference frame
        frame_num = self._frame_num
        video_placeholder = torch.zeros(
            1, cond_image.shape[1], frame_num - cond_image.shape[2],
            self._height, self._width,
            device=device, dtype=torch.bfloat16,
        )
        padding_frames = torch.cat([cond_image, video_placeholder], dim=2)
        with torch.no_grad():
            y = self._vae.encode(padding_frames).to(device).unsqueeze(0)
        self._msk = self._fn_get_msk(frame_num, cond_image, self.VAE_STRIDE, device)
        self._y = torch.cat([self._msk, y], dim=1)

        # Ref target masks
        self._ref_target_masks = torch.ones(
            3, self._height // self.VAE_STRIDE[1],
            self._width // self.VAE_STRIDE[2],
            device=device, dtype=torch.bfloat16,
        )

        # T5 encode default prompt
        t5_dev = "cpu" if self._t5_cpu else device
        with torch.no_grad():
            self._context = [
                self._text_encoder(texts=self._default_prompt, device=t5_dev)[0]
                .to(device, dtype=torch.bfloat16)
            ]

        # Reset streaming state for new identity
        self._reset_streaming_state()
        self._avatar_initialized = True

    # ── Streaming generation ──────────────────────────────────────────────

    async def generate_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[VideoChunk]:
        async for audio_chunk in audio_stream:
            if self._world_size > 1:
                for vc in self._generate_chunks_sync(audio_chunk):
                    yield vc
                    await asyncio.sleep(0)
            else:
                import queue as _queue
                loop = asyncio.get_running_loop()
                q: _queue.SimpleQueue = _queue.SimpleQueue()

                def _produce() -> None:
                    for vc in self._generate_chunks_sync(audio_chunk):
                        q.put(vc)
                    q.put(None)

                fut = loop.run_in_executor(None, _produce)
                while True:
                    vc = await loop.run_in_executor(None, q.get)
                    if vc is None:
                        break
                    yield vc
                await fut

    def _generate_chunks_sync(self, audio_chunk: AudioChunk) -> Iterator[VideoChunk]:
        with self._lock:
            try:
                if not self._avatar_initialized:
                    logger.warning("LiveAct avatar not initialized, skipping")
                    return

                # Decode incoming audio to 16kHz float32
                tgt_sr = 16000
                src_sr = int(audio_chunk.sample_rate or tgt_sr)
                audio_np = _audio_bytes_to_float32_mono(
                    audio_chunk.data, audio_chunk.format
                )
                audio_np = _resample_linear_mono(audio_np, src_sr, tgt_sr)

                if audio_np.size > 0:
                    if self._raw_audio.size == 0:
                        self._raw_audio = audio_np
                    else:
                        self._raw_audio = np.concatenate([self._raw_audio, audio_np])

                generated_any = False
                while self._can_generate_next():
                    generated_any = True
                    chunk_start = time.perf_counter()
                    frames = self._run_one_iteration_distributed(is_final=False)
                    elapsed = time.perf_counter() - chunk_start

                    if frames is None:
                        continue

                    self._chunk_counter += 1
                    nf, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
                    logger.info(
                        "LiveAct chunk: idx=%d frames=%d %dx%d fps=%d iter=%d elapsed=%.3fs is_final=%s",
                        self._chunk_counter, nf, w, h, self._fps,
                        self._iteration_count, elapsed, False,
                    )
                    yield VideoChunk(
                        frames=frames,
                        fps=self._fps,
                        chunk_index=self._chunk_counter,
                        is_final=False,
                    )

                if audio_chunk.is_final and self._has_pending_audio_for_iteration():
                    generated_any = True
                    chunk_start = time.perf_counter()
                    frames = self._run_one_iteration_distributed(is_final=True)
                    elapsed = time.perf_counter() - chunk_start

                    if frames is not None:
                        self._chunk_counter += 1
                        nf, h, w = frames.shape[0], frames.shape[1], frames.shape[2]
                        logger.info(
                            "LiveAct chunk: idx=%d frames=%d %dx%d fps=%d iter=%d elapsed=%.3fs is_final=%s",
                            self._chunk_counter, nf, w, h, self._fps,
                            self._iteration_count, elapsed, True,
                        )
                        yield VideoChunk(
                            frames=frames,
                            fps=self._fps,
                            chunk_index=self._chunk_counter,
                            is_final=True,
                        )

                if not generated_any:
                    return

            except Exception:
                logger.exception("LiveAct inference failed")

    def _iteration_audio_window(self, iteration: int) -> tuple[int, int]:
        """Return the absolute [start, end) audio sample window at 16 kHz."""
        fps = self._fps
        if iteration == 0:
            audio_start_frame = 0
            audio_end_frame = self._frame_num
        else:
            audio_start_frame = (iteration - 1) * self.BLKSZ_LST[-1] * self.VAE_STRIDE[0]
            audio_end_frame = audio_start_frame + self._frame_num

        sample_start = int(16000 * (audio_start_frame / fps))
        sample_end = int(16000 * ((audio_end_frame + 2) / fps))
        return sample_start, sample_end

    def _buffer_available_until(self) -> int:
        return self._raw_audio_start_sample + int(self._raw_audio.shape[0])

    def _has_pending_audio_for_iteration(self) -> bool:
        sample_start, _ = self._iteration_audio_window(self._iteration_count)
        return self._buffer_available_until() > sample_start

    def _can_generate_next(self) -> bool:
        """Check if we have enough accumulated audio for the next iteration."""
        iteration = self._iteration_count
        _, audio_end_sample = self._iteration_audio_window(iteration)
        return self._buffer_available_until() >= audio_end_sample

    def _prepare_iteration_audio_slice(self, iteration: int, is_final: bool) -> np.ndarray:
        """Prepare the 16 kHz audio slice for one iteration on rank 0."""
        del is_final  # final chunks are handled by zero-padding below
        sample_start_abs, sample_end_abs = self._iteration_audio_window(iteration)

        available_abs = self._buffer_available_until()
        if available_abs < sample_end_abs:
            pad_len = sample_end_abs - available_abs
            self._raw_audio = np.concatenate([
                self._raw_audio, np.zeros(pad_len, dtype=np.float32)
            ])

        sample_start = sample_start_abs - self._raw_audio_start_sample
        sample_end = sample_end_abs - self._raw_audio_start_sample
        return self._raw_audio[sample_start:sample_end]

    def _run_one_iteration_distributed(self, is_final: bool = False) -> np.ndarray | None:
        """Run one iteration; in distributed mode all ranks enter the same forward."""
        iteration = self._iteration_count
        audio_slice = self._prepare_iteration_audio_slice(iteration, is_final)

        if self._world_size <= 1:
            frames = self._run_one_iteration_local(audio_slice, iteration)
        else:
            frames = self._broadcast_and_run_iteration(audio_slice, iteration)

        self._trim_consumed_audio()
        return frames

    def _run_one_iteration_local(
        self, audio_slice: np.ndarray, iteration: int, return_frames: bool = True
    ) -> np.ndarray | None:
        """Run one diffusion iteration on the current rank."""
        import torchaudio
        import torchaudio.transforms as T

        device = self._device
        fps = self._fps
        frame_num = self._frame_num

        # Compute audio window in frame units.
        if iteration == 0:
            audio_start_idx = 0
            audio_end_idx = frame_num
        else:
            audio_start_idx = (iteration - 1) * self.BLKSZ_LST[-1] * self.VAE_STRIDE[0]
            audio_end_idx = audio_start_idx + frame_num

        sr_ori = 16000
        audio_tensor = torch.from_numpy(audio_slice).unsqueeze(0)  # [1, samples]

        # Tempo adjust + resample (matching generate.py)
        rate = 25.0 / fps
        audio_resampled, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_tensor, sr_ori, [["tempo", f"{rate}"]]
        )
        resampler = T.Resample(sr_ori, 16000)
        audio_resampled = resampler(audio_resampled) * 3.0

        # Wav2Vec2 encode
        audio_embedding = self._fn_get_embedding(
            audio_resampled[0], self._wav2vec_fe, self._audio_encoder, device=device,
        )
        # Since we extracted just the window, use indices from 0
        audio_embs = self._fn_get_audio_emb(audio_embedding, 0, frame_num, device)

        # Determine block index
        f_idx = 0 if iteration == 0 else 1

        # y_cut
        y_cut = self._y[:, :, :frame_num // 4 + 1, ...]

        lat_h = self._height // self.VAE_STRIDE[1]
        lat_w = self._width // self.VAE_STRIDE[2]

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            latent = torch.randn(
                16, self.BLKSZ_LST[f_idx], lat_h, lat_w,
                dtype=torch.bfloat16, device=device,
            )

            for i in range(len(self._timesteps) - 1):
                arg_c = {
                    "context": self._context,
                    "clip_fea": self._clip_context,
                    "ref_target_masks": self._ref_target_masks,
                    "audio": audio_embs,
                    "y": y_cut[:, :, sum(self.BLKSZ_LST[:f_idx]):sum(self.BLKSZ_LST[:f_idx + 1])],
                    "start_idx": sum(self.BLKSZ_LST[:f_idx]) * self._frame_len,
                    "end_idx": sum(self.BLKSZ_LST[:f_idx + 1]) * self._frame_len,
                    "update_cache": iteration > 1,
                }
                noise_pred = self._wan_model(
                    [latent.to(device)], t=self._timesteps[i],
                    kv_cache=self._kv_cache[i],
                    skip_audio=i not in (1, 2),
                    **arg_c,
                )[0]

                # Audio CFG
                if self._audio_cfg > 1.0 and i in (1, 2):
                    arg_null = dict(arg_c)
                    arg_null["audio"] = torch.zeros_like(audio_embs)
                    noise_null = self._wan_model(
                        [latent.to(device)], t=self._timesteps[i],
                        kv_cache=self._kv_cache[i],
                        skip_audio=i not in (1, 2),
                        **arg_null,
                    )[0]
                    noise_pred = noise_null + self._audio_cfg * (noise_pred - noise_null)

                dt = (self._timesteps[i] - self._timesteps[i + 1]) / 1000
                latent = latent + (-noise_pred) * dt[0]

            # VAE decode with overlap
            if iteration == 0:
                videos = self._vae.decode(latent)
            else:
                combined = torch.cat([self._pre_latent[:, -3:], latent], dim=1)
                videos = self._vae.decode(combined)[:, :, 9:]

            self._pre_latent = latent
            self._iteration_count += 1

        if not return_frames:
            return None

        # Convert to numpy uint8 (N, H, W, 3)
        frames = (
            (videos.squeeze(0).permute(1, 2, 3, 0) + 1.0) * 127.5
        ).clamp(0, 255).to(torch.uint8).cpu().numpy()

        return frames

    def _broadcast_and_run_iteration(
        self, audio_slice: np.ndarray, iteration: int
    ) -> np.ndarray | None:
        """Rank 0 broadcasts the infer command and audio payload to all ranks."""
        if self._world_size <= 1:
            return self._run_one_iteration_local(audio_slice, iteration)
        if self._rank != 0:
            return None

        cuda_dev = torch.device(f"cuda:{self._rank}")
        audio_np = np.asarray(audio_slice, dtype=np.float32)
        with self._dist_control_lock:
            self._broadcast_dist_cmd_locked(
                _DIST_OP_INFER,
                int(audio_np.shape[0]),
                int(iteration),
                0,
            )
            payload = torch.from_numpy(audio_np).to(cuda_dev, non_blocking=False)
            dist.broadcast(payload, src=0)
            self._note_dist_command_locked()
        return self._run_one_iteration_local(audio_np, iteration)

    def _trim_consumed_audio(self) -> None:
        """Drop audio prefix that future iterations no longer need."""
        next_start_abs, _ = self._iteration_audio_window(self._iteration_count)
        trim = next_start_abs - self._raw_audio_start_sample
        if trim <= 0:
            return
        trim = min(trim, int(self._raw_audio.shape[0]))
        self._raw_audio = self._raw_audio[trim:]
        self._raw_audio_start_sample += trim

    # ── Reset ─────────────────────────────────────────────────────────────

    async def reset(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._reset_sync)

    def _reset_sync(self) -> None:
        with self._lock:
            self._reset_sync_local()
            self._distributed_reset_if_needed()

    def _reset_sync_local(self) -> None:
        self._reset_streaming_state()

    def get_fps(self) -> int:
        return self._fps

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        if self._world_size > 1 and self._rank == 0:
            self._stop_dist_keepalive_if_needed()
            self._distributed_shutdown_if_needed()
            time.sleep(0.2)

        self._dist_worker_stop.set()
        if self._dist_worker_thread is not None and self._dist_worker_thread.is_alive():
            self._dist_worker_thread.join(timeout=5.0)

        if dist.is_initialized():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                logger.exception("destroy_process_group failed")

        self._cleanup()

    def _cleanup(self) -> None:
        self._dist_worker_thread = None
        self._dist_keepalive_thread = None
        self._wan_model = None
        self._vae = None
        self._clip = None
        self._text_encoder = None
        self._audio_encoder = None
        self._wav2vec_fe = None
        self._kv_cache = None
        self._clip_context = None
        self._y = None
        self._context = None
        self._avatar_initialized = False

        if self._default_avatar_path and self._default_avatar_is_temp:
            try:
                os.unlink(self._default_avatar_path)
            except Exception:
                pass
        self._default_avatar_path = None
        self._default_avatar_is_temp = False

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # ── Distributed ───────────────────────────────────────────────────────

    def _start_dist_worker_if_needed(self) -> None:
        if self._world_size <= 1 or self._rank == 0:
            return
        if self._dist_worker_thread is not None and self._dist_worker_thread.is_alive():
            return
        self._dist_worker_stop.clear()
        self._dist_worker_thread = threading.Thread(
            target=self._dist_worker_loop,
            name=f"liveact-dist-worker-rank{self._rank}",
            daemon=True,
        )
        self._dist_worker_thread.start()

    def _start_dist_keepalive_if_needed(self) -> None:
        if self._world_size <= 1 or self._rank != 0:
            return
        if self._dist_keepalive_thread is not None and self._dist_keepalive_thread.is_alive():
            return
        self._dist_keepalive_stop.clear()
        self._dist_last_command_monotonic = time.monotonic()
        self._dist_keepalive_thread = threading.Thread(
            target=self._dist_keepalive_loop,
            name="liveact-dist-keepalive",
            daemon=True,
        )
        self._dist_keepalive_thread.start()
        logger.info(
            "LiveAct dist keepalive started: interval=%.1fs idle=%.1fs",
            self._dist_keepalive_interval_s,
            self._dist_keepalive_idle_s,
        )

    def _stop_dist_keepalive_if_needed(self) -> None:
        self._dist_keepalive_stop.set()
        if self._dist_keepalive_thread is not None and self._dist_keepalive_thread.is_alive():
            self._dist_keepalive_thread.join(timeout=5.0)

    def _dist_keepalive_loop(self) -> None:
        while not self._dist_keepalive_stop.wait(self._dist_keepalive_interval_s):
            if not dist.is_initialized():
                continue
            with self._lock:
                if self._dist_keepalive_stop.is_set():
                    break
                idle_for = time.monotonic() - self._dist_last_command_monotonic
                if idle_for < self._dist_keepalive_idle_s:
                    continue
                with self._dist_control_lock:
                    self._broadcast_dist_cmd_locked(_DIST_OP_KEEPALIVE)

    def _broadcast_dist_cmd_locked(
        self,
        op_code: int,
        param1: int = 0,
        param2: int = 0,
        param3: int = 0,
    ) -> None:
        if self._world_size <= 1 or self._rank != 0 or not dist.is_initialized():
            return
        cuda_dev = torch.device(f"cuda:{self._rank}")
        cmd = torch.tensor(
            [int(op_code), int(param1), int(param2), int(param3)],
            dtype=torch.int32,
            device=cuda_dev,
        )
        dist.broadcast(cmd, src=0)
        self._note_dist_command_locked()

    def _note_dist_command_locked(self) -> None:
        self._dist_last_command_monotonic = time.monotonic()

    def _dist_worker_loop(self) -> None:
        """Worker loop for non-rank-0 processes in distributed mode.

        Command protocol (tensor-based):
          cmd_tensor = [op_code, param1, param2, param3]
            op_code 0: infer (param1=audio_len, param2=iteration, audio data follows)
            op_code 1: shutdown
            op_code 2: reset
            op_code 3: keepalive
            op_code 4: set_avatar (param1=image_bytes_len)
        """
        if self._world_size <= 1:
            return
        logger.info("LiveAct dist worker started: rank=%d/%d", self._rank, self._world_size)

        try:
            while not self._dist_worker_stop.is_set():
                cuda_dev = torch.device(f"cuda:{self._rank}")
                cmd = torch.zeros(4, dtype=torch.int32, device=cuda_dev)
                dist.broadcast(cmd, src=0)
                op = int(cmd[0].item())

                if op == _DIST_OP_SHUTDOWN:
                    break
                if op == _DIST_OP_RESET:
                    self._reset_sync_local()
                    continue
                if op == _DIST_OP_KEEPALIVE:
                    continue
                if op == _DIST_OP_SET_AVATAR:
                    img_len = int(cmd[1].item())
                    recv = torch.empty(img_len, dtype=torch.uint8, device=cuda_dev)
                    dist.broadcast(recv, src=0)
                    img_bytes = recv.cpu().numpy().tobytes()
                    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp_path = tmp.name
                    try:
                        tmp.write(img_bytes)
                        tmp.close()
                        self._set_avatar_sync_local(tmp_path)
                    except Exception:
                        logger.exception("LiveAct dist set_avatar failed: rank=%d", self._rank)
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    continue
                if op == _DIST_OP_INFER:
                    # Receive audio data and run the same iteration
                    audio_len = int(cmd[1].item())
                    iteration = int(cmd[2].item())
                    if audio_len <= 0:
                        continue
                    recv = torch.empty(audio_len, dtype=torch.float32, device=cuda_dev)
                    dist.broadcast(recv, src=0)
                    if iteration != self._iteration_count:
                        raise RuntimeError(
                            "LiveAct worker iteration mismatch: "
                            f"local={self._iteration_count} broadcast={iteration}"
                        )
                    audio_np = recv.detach().cpu().numpy().astype(np.float32, copy=False)
                    _ = self._run_one_iteration_local(
                        audio_np, iteration, return_frames=False
                    )
                    continue

        except Exception:
            logger.exception("LiveAct dist worker crashed: rank=%d", self._rank)
        logger.info("LiveAct dist worker stopped: rank=%d", self._rank)

    def _distributed_set_avatar(self, image_path: str) -> None:
        cuda_dev = torch.device(f"cuda:{self._rank}")
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        if not img_bytes:
            raise ValueError(f"Avatar image file is empty: {image_path}")

        with self._dist_control_lock:
            self._broadcast_dist_cmd_locked(_DIST_OP_SET_AVATAR, len(img_bytes), 0, 0)
            img_tensor = torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8).to(cuda_dev)
            dist.broadcast(img_tensor, src=0)
            self._note_dist_command_locked()
        self._set_avatar_sync_local(image_path)

    def _distributed_reset_if_needed(self) -> None:
        if self._world_size <= 1 or self._rank != 0:
            return
        with self._dist_control_lock:
            self._broadcast_dist_cmd_locked(_DIST_OP_RESET)

    def _distributed_shutdown_if_needed(self) -> None:
        if self._world_size <= 1 or self._rank != 0:
            return
        with self._dist_control_lock:
            self._broadcast_dist_cmd_locked(_DIST_OP_SHUTDOWN)
