import asyncio
import logging
import os
import threading
import tempfile
import time
from collections import deque
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


def _ensure_distributed_env_for_world_size(world_size: int) -> None:
    """Validate minimal env vars for torch.distributed launch."""
    if world_size <= 1:
        return

    required = ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            "FlashHead world_size>1 requires distributed launch env vars. "
            f"Missing: {', '.join(missing)}. "
            "Use torchrun and set world_size to match WORLD_SIZE."
        )

    env_world_size = int(os.environ["WORLD_SIZE"])
    if env_world_size != int(world_size):
        raise RuntimeError(
            "FlashHead world_size mismatch: "
            f"config={world_size}, WORLD_SIZE={env_world_size}. "
            "Please keep them consistent."
        )


def _apply_cuda_visible_devices(config: PluginConfig) -> None:
    """Apply CUDA_VISIBLE_DEVICES from plugin config if provided."""
    raw = config.params.get("cuda_visible_devices")
    if raw is None:
        return
    value = str(raw).strip()
    if not value:
        raise ValueError("cuda_visible_devices is set but empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    logger.info("FlashHead using CUDA_VISIBLE_DEVICES=%s", value)


def _distributed_all_ranks_ready(local_ready: bool) -> bool:
    """Synchronize avatar init readiness across ranks before warmup/worker loop."""
    if not dist.is_available() or not dist.is_initialized():
        return local_ready

    device_index = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    ready = torch.tensor(
        [1 if local_ready else 0],
        dtype=torch.int32,
        device=torch.device(f"cuda:{device_index}") if torch.cuda.is_available() else "cpu",
    )
    dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    return bool(int(ready.item()))


class FlashHeadAvatarPlugin(AvatarPlugin):
    """Wraps existing FlashHead inference code as an Avatar plugin.

    Key design:
    - Maintains 8s audio sliding window (deque)
    - Keeps latent_motion_frames continuity between chunks
    - Thread lock for GPU serialization
    - All flash_head imports are cached as instance attributes (no repeated imports)
    """

    name = "avatar.flash_head"

    def __init__(self) -> None:
        self.pipeline = None
        self.infer_params: dict = {}
        self._world_size: int = 1
        self.audio_deque: deque | None = None
        self._lock = threading.Lock()
        self._chunk_counter = 0
        self._avatar_initialized = False
        self._default_avatar_path: str | None = None
        self._default_avatar_is_temp: bool = False
        # Cached function references (set during _init_sync)
        self._fn_get_base_data = None
        self._fn_get_audio_embedding = None
        self._fn_run_pipeline = None
        # Audio accumulation: only run the model when we have at least one
        # full "slice" of audio (= net_output_frames samples at target SR).
        # Keep an explicit pending audio buffer and consume exactly one slice
        # per generated chunk so produced video duration matches consumed audio.
        self._pending_audio: np.ndarray = np.array([], dtype=np.float32)
        self._slice_len_samples: int = 0  # set in _init_audio_deque
        self._rank: int = int(os.environ.get("RANK", "0"))
        self._world_size_env: int = int(os.environ.get("WORLD_SIZE", "1"))
        self._dist_worker_thread: threading.Thread | None = None
        self._dist_worker_stop = threading.Event()

    async def initialize(self, config: PluginConfig) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._init_sync, config)

    def _create_default_avatar_placeholder(self) -> tuple[str, bool]:
        height = int(self.infer_params["height"])
        width = int(self.infer_params["width"])
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()
        img = Image.new("RGB", (width, height), color=(128, 128, 128))
        img.save(tmp_path, format="PNG")
        return tmp_path, True

    def _init_sync(self, config: PluginConfig) -> None:
        _apply_cuda_visible_devices(config)

        world_size = int(config.params.get("world_size", 1))
        if world_size < 1:
            raise ValueError(f"Invalid world_size={world_size}, expected >= 1")
        self._world_size = world_size
        _ensure_distributed_env_for_world_size(world_size)
        warmup_policy = resolve_avatar_warmup_policy(
            config,
            world_size=self._world_size,
        )

        # Only manipulate sys.path if models_dir is provided (local vendor code)
        models_dir = config.params.get("models_dir")
        if models_dir:
            import sys
            resolved = str(Path(models_dir).resolve())
            with _sys_path_lock:
                if resolved not in sys.path:
                    sys.path.insert(0, resolved)

        # Import once and cache all functions
        from flash_head.inference import (
            configure_infer_params,
            configure_runtime_options,
            get_pipeline,
            get_infer_params,
            get_base_data,
            get_audio_embedding,
            run_pipeline,
        )

        configure_runtime_options(config.params)
        configure_infer_params(config.params.get("infer_params"))
        self._fn_get_base_data = get_base_data
        self._fn_get_audio_embedding = get_audio_embedding
        self._fn_run_pipeline = run_pipeline

        self.pipeline = get_pipeline(
            world_size=self._world_size,
            ckpt_dir=config.params["checkpoint_dir"],
            model_type=config.params.get("model_type", "lite"),
            wav2vec_dir=config.params["wav2vec_dir"],
        )
        self.infer_params = get_infer_params()
        self._init_audio_deque()
        self._chunk_counter = 0
        # Some environments deadlock when torch.distributed collectives are
        # executed from Python background threads. Allow running the
        # distributed worker loop on the main thread for non-rank0.
        self._dist_worker_main_thread = (
            _parse_bool(
                os.environ.get(
                    "FLASHHEAD_DIST_WORKER_MAIN_THREAD",
                    config.params.get("dist_worker_main_thread"),
                ),
                default=False,
            )
        )

        # Use a gray placeholder avatar for initialization and warmup.
        base_seed = int(config.params.get("seed", 9999))
        try:
            image_path, is_temp_image = self._create_default_avatar_placeholder()
            self._default_avatar_path = image_path
            self._default_avatar_is_temp = is_temp_image

            self._fn_get_base_data(
                self.pipeline,
                image_path,
                base_seed=base_seed,
                use_face_crop=False,
            )
            self._avatar_initialized = True
            device = getattr(self.pipeline, "device", None)
            logger.info(
                "FlashHead avatar loaded: model_type=%s checkpoint=%s wav2vec=%s "
                "default_avatar=%s seed=%s use_face_crop=%s device=%s world_size=%s",
                config.params.get("model_type", "lite"),
                config.params["checkpoint_dir"],
                config.params["wav2vec_dir"],
                image_path,
                base_seed,
                False,
                device,
                self._world_size,
            )
        except Exception:
            logger.exception("FlashHead pipeline default avatar init failed")
            self._avatar_initialized = False

        avatar_ready = _distributed_all_ranks_ready(self._avatar_initialized)

        if (
            self._world_size > 1
            and self._rank != 0
        ):
            if self._dist_worker_main_thread:
                # Block here to participate in collectives on the main thread
                # only after all ranks finished default avatar/base-data setup.
                self._dist_worker_loop()
                return

            self._start_dist_worker_if_needed()
            return

        if avatar_ready and warmup_policy.enabled:
            self._warmup()
        elif self._rank == 0:
            logger.info(
                "FlashHead warmup skipped: avatar_ready=%s global_enabled=%s distributed_enabled=%s world_size=%d",
                avatar_ready,
                warmup_policy.global_enabled,
                warmup_policy.distributed_enabled,
                self._world_size,
            )

    def _start_dist_worker_if_needed(self) -> None:
        if self._world_size <= 1 or self._rank == 0:
            return
        if self._dist_worker_thread is not None and self._dist_worker_thread.is_alive():
            return
        self._dist_worker_stop.clear()
        self._dist_worker_thread = threading.Thread(
            target=self._dist_worker_loop,
            name=f"flashhead-dist-worker-rank{self._rank}",
            daemon=True,
        )
        self._dist_worker_thread.start()

    def _dist_worker_loop(self) -> None:
        if self._world_size <= 1:
            return
        logger.info(
            "FlashHead distributed worker loop started: rank=%d/%d",
            self._rank,
            self._world_size,
        )
        # Distributed command protocol (tensor-based) to avoid
        # `dist.broadcast_object_list` deadlocks.
        # cmd_tensor = [op_code, param1, param2, param3]
        #   op_code: 0=infer (param1=audio_len, param2=start_idx, param3=end_idx)
        #            1=shutdown, 2=reset
        #            4=set_avatar (param1=image_bytes_len, param2=use_face_crop)
        try:
            while not self._dist_worker_stop.is_set():
                cuda_device = torch.device(f"cuda:{self._rank}")
                cmd_tensor = torch.zeros(
                    4, dtype=torch.int32, device=cuda_device
                )
                dist.broadcast(cmd_tensor, src=0)
                op_code = int(cmd_tensor[0].item())
                audio_len = int(cmd_tensor[1].item())
                audio_start_idx = int(cmd_tensor[2].item())
                audio_end_idx = int(cmd_tensor[3].item())
                logger.debug(
                    "FlashHead dist worker got cmd: rank=%d op_code=%d audio_len=%d start_idx=%d end_idx=%d",
                    self._rank,
                    op_code,
                    audio_len,
                    audio_start_idx,
                    audio_end_idx,
                )

                if op_code == 1:
                    break
                if op_code == 2:
                    self._reset_sync_local_only()
                    continue
                if op_code == 4:
                    # set_avatar: receive image bytes, then call
                    # get_base_data so vae.encode's all_gather matches rank 0.
                    image_len = int(cmd_tensor[1].item())
                    use_face_crop = bool(cmd_tensor[2].item())
                    recv_img = torch.empty(
                        image_len, dtype=torch.uint8, device=cuda_device
                    )
                    dist.broadcast(recv_img, src=0)
                    image_bytes = recv_img.cpu().numpy().tobytes()
                    tmp = tempfile.NamedTemporaryFile(
                        suffix=".png", delete=False
                    )
                    tmp_path = tmp.name
                    try:
                        tmp.write(image_bytes)
                        tmp.close()
                        self._set_avatar_sync_local_only(
                            tmp_path, use_face_crop
                        )
                        logger.debug(
                            "FlashHead dist worker set_avatar done: rank=%d image_bytes=%d",
                            self._rank,
                            image_len,
                        )
                    except Exception:
                        logger.exception(
                            "FlashHead dist worker set_avatar failed: rank=%d",
                            self._rank,
                        )
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    continue
                if op_code != 0:
                    continue

                if audio_len <= 0:
                    continue
                recv = torch.empty(
                    audio_len, dtype=torch.float32, device=cuda_device
                )
                dist.broadcast(recv, src=0)
                audio_array = recv.detach().cpu().numpy().astype(np.float64, copy=False)
                audio_embedding = self._fn_get_audio_embedding(
                    self.pipeline, audio_array, audio_start_idx, audio_end_idx
                )
                _ = self._fn_run_pipeline(self.pipeline, audio_embedding)
        except Exception:
            logger.exception(
                "FlashHead distributed worker loop crashed: rank=%d world_size=%d",
                self._rank,
                self._world_size,
            )
        logger.info("FlashHead distributed worker loop stopped: rank=%d", self._rank)

    def _distributed_set_avatar(self, image_path: str, use_face_crop: bool) -> None:
        """Broadcast avatar image to all ranks, then all ranks call get_base_data simultaneously.

        get_base_data -> prepare_params -> vae.encode uses dist.all_gather,
        so every rank must enter it at the same time.  We use tensor-based
        broadcast (not broadcast_object_list which has been unreliable).
        """
        cuda_device = torch.device(f"cuda:{self._rank}")

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        if not image_bytes:
            raise ValueError(f"Avatar image file is empty: {image_path}")

        # op_code=4: set_avatar
        cmd_tensor = torch.tensor(
            [4, len(image_bytes), int(use_face_crop), 0],
            dtype=torch.int32,
            device=cuda_device,
        )
        dist.broadcast(cmd_tensor, src=0)

        image_tensor = torch.frombuffer(
            bytearray(image_bytes), dtype=torch.uint8
        ).to(cuda_device)
        dist.broadcast(image_tensor, src=0)

        logger.debug(
            "FlashHead distributed set_avatar broadcast: rank=%d image_bytes=%d use_face_crop=%s",
            self._rank,
            len(image_bytes),
            use_face_crop,
        )

        # Now rank 0 calls get_base_data while workers do the same from
        # _dist_worker_loop.  vae.encode's all_gather matches across ranks.
        self._set_avatar_sync_local_only(image_path, use_face_crop)

    def _distributed_reset_if_needed(self) -> None:
        if self._world_size <= 1:
            return
        if self._rank != 0:
            return
        cmd_tensor = torch.tensor(
            [2, 0, 0, 0], dtype=torch.int32, device=torch.device(f"cuda:{self._rank}")
        )
        dist.broadcast(cmd_tensor, src=0)

    def _distributed_shutdown_if_needed(self) -> None:
        if self._world_size <= 1:
            return
        if self._rank != 0:
            return
        cmd_tensor = torch.tensor(
            [1, 0, 0, 0], dtype=torch.int32, device=torch.device(f"cuda:{self._rank}")
        )
        dist.broadcast(cmd_tensor, src=0)

    def _run_pipeline_distributed(self, audio_array: np.ndarray, audio_start_idx: int, audio_end_idx: int):
        if self._world_size <= 1:
            audio_embedding = self._fn_get_audio_embedding(
                self.pipeline, audio_array, audio_start_idx, audio_end_idx
            )
            return self._fn_run_pipeline(self.pipeline, audio_embedding)
        # Rank0 broadcasts infer command and audio window, then all ranks run
        # the same get_audio_embedding + run_pipeline path.
        if self._rank != 0:
            return None
        audio_np = np.asarray(audio_array, dtype=np.float32)
        logger.debug(
            "FlashHead dist broadcast infer start: rank=%d audio_len=%d start_idx=%d end_idx=%d world_size=%d",
            self._rank,
            int(audio_np.shape[0]),
            audio_start_idx,
            audio_end_idx,
            self._world_size,
        )
        cmd_tensor = torch.tensor(
            [
                0,  # op_code: infer
                int(audio_np.shape[0]),
                int(audio_start_idx),
                int(audio_end_idx),
            ],
            dtype=torch.int32,
            device=torch.device(f"cuda:{self._rank}"),
        )
        dist.broadcast(cmd_tensor, src=0)
        cuda_device = torch.device(f"cuda:{self._rank}")
        payload = torch.from_numpy(audio_np).to(cuda_device, non_blocking=False)
        dist.broadcast(payload, src=0)
        audio_embedding = self._fn_get_audio_embedding(
            self.pipeline, audio_np.astype(np.float64, copy=False), audio_start_idx, audio_end_idx
        )
        return self._fn_run_pipeline(self.pipeline, audio_embedding)

    def _init_audio_deque(self) -> None:
        sr = self.infer_params["sample_rate"]
        duration = self.infer_params["cached_audio_duration"]
        maxlen = sr * duration
        self.audio_deque = deque(maxlen=maxlen)
        self.audio_deque.extend(np.zeros(maxlen, dtype=np.float64))

        # Compute the minimum audio accumulation threshold.
        # The model generates frame_num frames total; the first motion_frames_num
        # are discarded as temporal context, leaving net_frames of actual output.
        # Gradio reference: slice_len = frame_num - motion_frames_num,
        #   slice_len_samples = slice_len * sample_rate // tgt_fps
        # We use the same formula so each model call processes exactly one output
        # chunk's worth of new audio (matching the Gradio streaming approach).
        frame_num = int(self.infer_params.get("frame_num", 33))
        motion_frames_num = int(self.infer_params.get("motion_frames_num", 5))
        tgt_fps = int(self.infer_params.get("tgt_fps", 25))
        net_frames = frame_num - motion_frames_num  # e.g. 28
        self._slice_len_samples = net_frames * sr // tgt_fps  # e.g. 17920

        # Keep strict 1:1 mapping between consumed audio slice and generated
        # video chunk duration; do not reduce slice length by ratio.
        self._pending_audio = np.array([], dtype=np.float32)

    def _warmup(self) -> None:
        if not self._avatar_initialized or self.pipeline is None or self.audio_deque is None:
            logger.info("FlashHead warmup skipped: pipeline/avatar not initialized")
            return
        if self._slice_len_samples <= 0:
            logger.info(
                "FlashHead warmup skipped: invalid slice_len_samples=%d",
                self._slice_len_samples,
            )
            return
        if self._world_size > 1 and self._rank != 0:
            return

        ip = self.infer_params
        audio_end_idx = ip["cached_audio_duration"] * ip["tgt_fps"]
        audio_start_idx = audio_end_idx - ip["frame_num"]
        silent_slice = np.zeros(self._slice_len_samples, dtype=np.float64)
        self.audio_deque.extend(silent_slice)
        audio_array = np.array(self.audio_deque, dtype=np.float64)

        logger.info(
            "FlashHead warmup: running silent inference to prime pipeline (world_size=%d slice_len_samples=%d)",
            self._world_size,
            self._slice_len_samples,
        )
        start_time = time.perf_counter()
        video = self._run_pipeline_distributed(
            audio_array,
            audio_start_idx,
            audio_end_idx,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - start_time

        motion_frames = int(ip.get("motion_frames_num", 5))
        num_frames = 0
        if video is not None:
            num_frames = int(video[motion_frames:].shape[0])

        logger.info(
            "FlashHead warmup done on rank %d: %d frames @ %d fps elapsed=%.3fs",
            self._rank,
            num_frames,
            int(ip["tgt_fps"]),
            elapsed_s,
        )

        self._reset_sync_local_only()
        if self._world_size > 1:
            self._distributed_reset_if_needed()

    async def set_avatar(self, image_path: str, use_face_crop: bool = False) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._set_avatar_sync, image_path, use_face_crop)

    def _set_avatar_sync(self, image_path: str, use_face_crop: bool) -> None:
        with self._lock:
            if self._world_size > 1 and self._rank == 0:
                # In distributed mode, all ranks must call get_base_data
                # simultaneously because vae.encode uses dist.all_gather.
                # Calling _set_avatar_sync_local_only on rank 0 alone would
                # deadlock: rank 0 enters all_gather while workers are waiting
                # at dist.broadcast in _dist_worker_loop.
                self._distributed_set_avatar(image_path, use_face_crop)
            else:
                self._set_avatar_sync_local_only(image_path, use_face_crop)

    def _set_avatar_sync_local_only(self, image_path: str, use_face_crop: bool) -> None:
        self._fn_get_base_data(
            self.pipeline, image_path, base_seed=9999, use_face_crop=use_face_crop
        )
        self._avatar_initialized = True

    async def generate_stream_paired(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[tuple[AudioChunk, VideoChunk | None]]:
        async for audio_chunk in audio_stream:
            # Important: when running distributed inference, torch.distributed
            # collectives may deadlock or behave unexpectedly if invoked from
            # a background executor thread. Keep distributed execution on the
            # event-loop thread for consistent collective ordering.
            has_video = False
            if self._world_size > 1:
                # Iterate sync generator directly on event-loop thread.
                # Yield each chunk immediately so gRPC can stream it out,
                # then await asyncio.sleep(0) to let the event loop flush
                # the response before starting the next chunk's inference.
                for video_chunk in self._generate_chunks_sync(audio_chunk):
                    has_video = True
                    yield audio_chunk, video_chunk
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
                    has_video = True
                    yield audio_chunk, vc
                await fut
            if not has_video:
                yield audio_chunk, None

    async def generate_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[VideoChunk]:
        async for _, video_chunk in self.generate_stream_paired(audio_stream):
            if video_chunk is not None:
                yield video_chunk

    def _generate_chunk_sync(self, audio_chunk: AudioChunk) -> VideoChunk | None:
        last = None
        for chunk in self._generate_chunks_sync(audio_chunk):
            last = chunk
        return last

    def _generate_chunks_sync(self, audio_chunk: AudioChunk) -> Iterator[VideoChunk]:
        with self._lock:
            try:
                # If SetAvatar hasn't been called (or reset cleared state),
                # skip generation rather than crashing the whole stream.
                if self.pipeline is None or not hasattr(self.pipeline, "frame_num"):
                    logger.warning("FlashHead pipeline not prepared (frame_num missing)")
                    return
                if getattr(self.pipeline, "latent_motion_frames", None) is None:
                    logger.warning("FlashHead pipeline not prepared (latent_motion_frames missing)")
                    return

                logger.debug(
                    "FlashHead recv AudioChunk: rank=%d bytes=%d sample_rate=%d channels=%d format=%s is_final=%s timestamp_ms=%d duration_ms=%d",
                    self._rank,
                    len(audio_chunk.data),
                    int(audio_chunk.sample_rate or 0),
                    int(audio_chunk.channels or 0),
                    str(audio_chunk.format),
                    bool(audio_chunk.is_final),
                    int(audio_chunk.timestamp_ms or 0),
                    int(audio_chunk.duration_ms or 0),
                )

                tgt_sr = int(self.infer_params["sample_rate"])
                src_sr = int(audio_chunk.sample_rate or tgt_sr)
                audio_np = _audio_bytes_to_float32_mono(
                    audio_chunk.data, audio_chunk.format
                )
                audio_np = _resample_linear_mono(audio_np, src_sr, tgt_sr)

                if audio_np.size > 0:
                    if self._pending_audio.size == 0:
                        self._pending_audio = audio_np
                    else:
                        self._pending_audio = np.concatenate([self._pending_audio, audio_np])

                logger.debug(
                    "FlashHead recv audio: rank=%d pcm_bytes=%d src_sr=%d tgt_sr=%d audio_samples=%d pending_samples=%d slice_len_samples=%d is_final=%s",
                    self._rank,
                    len(audio_chunk.data),
                    src_sr,
                    tgt_sr,
                    len(audio_np),
                    int(self._pending_audio.shape[0]),
                    self._slice_len_samples,
                    audio_chunk.is_final,
                )

                ip = self.infer_params
                audio_end_idx = ip["cached_audio_duration"] * ip["tgt_fps"]
                audio_start_idx = audio_end_idx - ip["frame_num"]
                if self._slice_len_samples <= 0:
                    return

                to_generate: list[np.ndarray] = []
                while int(self._pending_audio.shape[0]) >= self._slice_len_samples:
                    one = self._pending_audio[: self._slice_len_samples]
                    self._pending_audio = self._pending_audio[self._slice_len_samples :]
                    to_generate.append(one)

                # Flush tail on final: pad to one full slice, matching generate_video.py.
                if audio_chunk.is_final and int(self._pending_audio.shape[0]) > 0:
                    tail = self._pending_audio
                    self._pending_audio = np.array([], dtype=np.float32)
                    pad_len = self._slice_len_samples - int(tail.shape[0])
                    if pad_len > 0:
                        tail = np.concatenate([tail, np.zeros(pad_len, dtype=tail.dtype)])
                    to_generate.append(tail)

                if not to_generate:
                    logger.debug(
                        "FlashHead skip inference: rank=%d pending_samples=%d < slice_len_samples=%d is_final=%s",
                        self._rank,
                        int(self._pending_audio.shape[0]),
                        self._slice_len_samples,
                        audio_chunk.is_final,
                    )
                    return

                for idx, consume_slice in enumerate(to_generate):
                    self.audio_deque.extend(consume_slice)
                    audio_array = np.array(self.audio_deque, dtype=np.float64)
                    logger.debug(
                        "FlashHead begin pipeline: rank=%d consume_samples=%d pending_after=%d slice_len_samples=%d is_final=%s world_size=%d",
                        self._rank,
                        int(consume_slice.shape[0]),
                        int(self._pending_audio.shape[0]),
                        self._slice_len_samples,
                        audio_chunk.is_final and idx == len(to_generate) - 1,
                        self._world_size,
                    )

                    chunk_start_time = time.perf_counter()
                    video = self._run_pipeline_distributed(
                        audio_array, audio_start_idx, audio_end_idx
                    )  # 生成视频帧
                    chunk_elapsed_s = time.perf_counter() - chunk_start_time
                    if video is None:
                        logger.debug(
                            "FlashHead run_pipeline returned None: rank=%d world_size=%d audio_start_idx=%d audio_end_idx=%d elapsed=%.3fs",
                            self._rank,
                            self._world_size,
                            audio_start_idx,
                            audio_end_idx,
                            chunk_elapsed_s,
                        )
                        continue

                    motion_frames = ip.get("motion_frames_num", 5)
                    video = video[motion_frames:]
                    frames = video.cpu().numpy()
                    frames = np.clip(frames, 0, 255).astype(np.uint8)

                    self._chunk_counter += 1
                    nf, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
                    is_last_final = audio_chunk.is_final and idx == len(to_generate) - 1
                    logger.info(
                        "FlashHead video chunk generated: chunk_index=%d num_frames=%d %dx%d fps=%d "
                        "consumed_samples=%d is_final=%s elapsed=%.3fs",
                        self._chunk_counter,
                        nf,
                        w,
                        h,
                        int(ip["tgt_fps"]),
                        int(consume_slice.shape[0]),
                        is_last_final,
                        chunk_elapsed_s,
                    )
                    yield VideoChunk(
                        frames=frames,
                        fps=ip["tgt_fps"],
                        chunk_index=self._chunk_counter,
                        is_final=is_last_final,
                    )
            except Exception:
                logger.exception("FlashHead inference failed")

    async def reset(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._reset_sync)

    def _reset_sync(self) -> None:
        with self._lock:
            self._reset_sync_local_only()
            self._distributed_reset_if_needed()

    def _reset_sync_local_only(self) -> None:
        if self.pipeline is not None:
            # Keep motion continuity seed material if base data was prepared.
            # Setting it to None makes FlashHead.generate() fail later.
            if getattr(self.pipeline, "ref_img_latent", None) is not None:
                self.pipeline.latent_motion_frames = (
                    self.pipeline.ref_img_latent[:, :1].clone()
                )
            else:
                self.pipeline.latent_motion_frames = None
        self._init_audio_deque()
        self._pending_audio = np.array([], dtype=np.float32)
        self._chunk_counter = 0

    def get_fps(self) -> int:
        return self.infer_params.get("tgt_fps", 25)

    async def shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._shutdown_sync)

    def _shutdown_sync(self) -> None:
        """Release distributed workers, destroy NCCL group, then drop pipeline refs."""
        if self._world_size <= 1:
            self._cleanup_pipeline_state()
            return

        # Rank 0 must broadcast so rank>0 threads blocked on broadcast_object_list can exit.
        if self._rank == 0:
            self._distributed_shutdown_if_needed()
            time.sleep(0.2)

        self._dist_worker_stop.set()
        if self._dist_worker_thread is not None and self._dist_worker_thread.is_alive():
            self._dist_worker_thread.join(timeout=5.0)

        if dist.is_initialized():
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            try:
                dist.destroy_process_group()
            except Exception:
                logger.exception("torch.distributed destroy_process_group failed")

        self._cleanup_pipeline_state()

    def _cleanup_pipeline_state(self) -> None:
        self._dist_worker_thread = None
        self.pipeline = None
        self.audio_deque = None
        self._fn_get_base_data = None
        self._fn_get_audio_embedding = None
        self._fn_run_pipeline = None
        self._avatar_initialized = False
        if self._default_avatar_path and self._default_avatar_is_temp:
            try:
                os.unlink(self._default_avatar_path)
            except Exception:
                pass
        self._default_avatar_path = None
        self._default_avatar_is_temp = False
