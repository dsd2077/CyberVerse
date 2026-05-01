"""Tests for FlashHeadAvatarPlugin using mocked FlashHead inference."""
import asyncio
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from inference.core.types import AudioChunk, PluginConfig, VideoChunk
from inference.plugins.avatar.flash_head_plugin import FlashHeadAvatarPlugin

FLASH_HEAD_INFER_PARAMS = {
    "frame_num": 33,
    "motion_frames_latent_num": 2,
    "tgt_fps": 25,
    "sample_rate": 16000,
    "sample_shift": 5,
    "color_correction_strength": 1.0,
    "cached_audio_duration": 8,
    "num_heads": 12,
    "height": 512,
    "width": 512,
}


@pytest.fixture
def plugin():
    """Create a FlashHeadAvatarPlugin with mocked internals (no GPU needed)."""
    p = FlashHeadAvatarPlugin()
    p.infer_params = {
        "sample_rate": 16000,
        "cached_audio_duration": 8,
        "tgt_fps": 25,
        "frame_num": 33,
        "motion_frames_num": 5,
    }
    p._init_audio_deque()
    p.pipeline = MagicMock()
    return p


def test_plugin_name():
    p = FlashHeadAvatarPlugin()
    assert p.name == "avatar.flash_head"


def test_get_fps(plugin):
    assert plugin.get_fps() == 25


def test_audio_deque_initialization(plugin):
    assert plugin.audio_deque is not None
    assert len(plugin.audio_deque) == 16000 * 8  # 8s @ 16kHz
    assert plugin.audio_deque.maxlen == 16000 * 8


def test_audio_deque_sliding_window(plugin):
    # Feed 1s of audio (16000 samples)
    new_audio = [1.0] * 16000
    plugin.audio_deque.extend(new_audio)
    # Deque should still be 8s long
    assert len(plugin.audio_deque) == 16000 * 8
    # Last 16000 samples should be 1.0
    arr = list(plugin.audio_deque)
    assert all(v == 1.0 for v in arr[-16000:])
    assert all(v == 0.0 for v in arr[:16000 * 7])


def _fake_flash_head_inference_module():
    package = types.ModuleType("flash_head")
    module = types.ModuleType("flash_head.inference")
    module.configure_infer_params = MagicMock()
    module.configure_runtime_options = MagicMock()
    module.get_pipeline = MagicMock(return_value=MagicMock(device="cpu"))
    module.get_infer_params = MagicMock(return_value={
        "sample_rate": 16000,
        "cached_audio_duration": 8,
        "tgt_fps": 25,
        "frame_num": 33,
        "motion_frames_num": 5,
        "height": 512,
        "width": 512,
    })
    module.get_base_data = MagicMock()
    module.get_audio_embedding = MagicMock()
    module.run_pipeline = MagicMock(return_value=torch.zeros((33, 1, 1, 3)))
    package.inference = module
    return package, module


@pytest.mark.asyncio
async def test_reset(plugin):
    plugin._chunk_counter = 10
    plugin.audio_deque.extend([1.0] * 1000)
    await plugin.reset()
    assert plugin._chunk_counter == 0
    assert all(v == 0.0 for v in plugin.audio_deque)
    plugin.pipeline.latent_motion_frames = None


@pytest.mark.asyncio
async def test_generate_stream_produces_video_chunks(plugin):
    """Test the full generate_stream flow with mocked inference."""
    fake_frames = np.zeros((28, 512, 512, 3), dtype=np.uint8)

    def mock_generate(audio_chunk):
        plugin._chunk_counter += 1
        yield VideoChunk(
            frames=fake_frames,
            fps=25,
            chunk_index=plugin._chunk_counter,
            is_final=audio_chunk.is_final,
        )

    plugin._generate_chunks_sync = mock_generate

    # Create async audio stream
    audio_data = np.zeros(17920, dtype=np.float32).tobytes()  # 1.12s chunk

    async def audio_stream():
        yield AudioChunk(data=audio_data, sample_rate=16000, is_final=False)
        yield AudioChunk(data=audio_data, sample_rate=16000, is_final=True)

    chunks = []
    async for vc in plugin.generate_stream(audio_stream()):
        chunks.append(vc)

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 1
    assert chunks[0].is_final is False
    assert chunks[1].chunk_index == 2
    assert chunks[1].is_final is True
    assert chunks[0].frames.shape == (28, 512, 512, 3)


@pytest.mark.asyncio
async def test_shutdown(plugin):
    await plugin.shutdown()
    assert plugin.pipeline is None
    assert plugin.audio_deque is None


def test_init_sync_runs_warmup_when_enabled():
    plugin = FlashHeadAvatarPlugin()
    config = PluginConfig(
        plugin_name="avatar.flash_head",
        params={
            "world_size": 1,
            "models_dir": "models",
            "checkpoint_dir": "/tmp/ckpt",
            "wav2vec_dir": "/tmp/wav2vec",
            "model_type": "lite",
            "compile_model": True,
            "compile_vae": True,
            "dist_worker_main_thread": True,
            "infer_params": FLASH_HEAD_INFER_PARAMS,
        },
        shared={
            "warmup": {
                "enabled": True,
                "distributed": {"enabled": True, "timeout_s": 30},
            }
        },
    )
    fake_package, fake_module = _fake_flash_head_inference_module()

    with patch.dict(
        sys.modules,
        {"flash_head": fake_package, "flash_head.inference": fake_module},
    ):
        with patch.object(
            plugin,
            "_create_default_avatar_placeholder",
            return_value=("/tmp/avatar.png", False),
        ):
            with patch.object(plugin, "_warmup") as warmup:
                plugin._init_sync(config)

    fake_module.configure_runtime_options.assert_called_once_with(config.params)
    fake_module.configure_infer_params.assert_called_once_with(FLASH_HEAD_INFER_PARAMS)
    warmup.assert_called_once_with()
    assert plugin._dist_worker_main_thread is True


def test_init_sync_skips_warmup_when_disabled():
    plugin = FlashHeadAvatarPlugin()
    config = PluginConfig(
        plugin_name="avatar.flash_head",
        params={
            "world_size": 1,
            "models_dir": "models",
            "checkpoint_dir": "/tmp/ckpt",
            "wav2vec_dir": "/tmp/wav2vec",
            "model_type": "lite",
            "compile_model": True,
            "compile_vae": True,
            "dist_worker_main_thread": True,
            "infer_params": FLASH_HEAD_INFER_PARAMS,
        },
        shared={
            "warmup": {
                "enabled": False,
                "distributed": {"enabled": True, "timeout_s": 30},
            }
        },
    )
    fake_package, fake_module = _fake_flash_head_inference_module()

    with patch.dict(
        sys.modules,
        {"flash_head": fake_package, "flash_head.inference": fake_module},
    ):
        with patch.object(
            plugin,
            "_create_default_avatar_placeholder",
            return_value=("/tmp/avatar.png", False),
        ):
            with patch.object(plugin, "_warmup") as warmup:
                plugin._init_sync(config)

    fake_module.configure_runtime_options.assert_called_once_with(config.params)
    fake_module.configure_infer_params.assert_called_once_with(FLASH_HEAD_INFER_PARAMS)
    warmup.assert_not_called()
    assert plugin._dist_worker_main_thread is True


def test_warmup_uses_pipeline_path_and_resets_transient_state(plugin):
    plugin._avatar_initialized = True
    plugin._pending_audio = np.ones(32, dtype=np.float32)
    plugin._chunk_counter = 9
    plugin.pipeline.ref_img_latent = torch.zeros((1, 2, 3), dtype=torch.float32)
    plugin.pipeline.latent_motion_frames = None
    plugin._run_pipeline_distributed = MagicMock(
        return_value=torch.zeros((33, 1, 1, 3), dtype=torch.float32)
    )

    plugin.audio_deque.extend(np.ones(plugin._slice_len_samples, dtype=np.float64))

    plugin._warmup()

    plugin._run_pipeline_distributed.assert_called_once()
    audio_array, audio_start_idx, audio_end_idx = (
        plugin._run_pipeline_distributed.call_args.args
    )
    assert isinstance(audio_array, np.ndarray)
    assert audio_start_idx == 167
    assert audio_end_idx == 200
    assert plugin._chunk_counter == 0
    assert plugin._pending_audio.size == 0
    assert np.count_nonzero(np.array(plugin.audio_deque, dtype=np.float64)) == 0
    assert plugin.pipeline.latent_motion_frames is not None


def test_generate_chunk_sync_audio_indexing(plugin):
    """Verify the audio indexing math for FlashHead."""
    ip = plugin.infer_params
    audio_end_idx = ip["cached_audio_duration"] * ip["tgt_fps"]  # 8*25=200
    audio_start_idx = audio_end_idx - ip["frame_num"]  # 200-33=167
    assert audio_start_idx == 167
    assert audio_end_idx == 200
    assert audio_end_idx - audio_start_idx == 33  # frame_num


def test_generation_started_attaches_trace_to_first_video_chunk(plugin):
    plugin._run_pipeline_distributed = MagicMock(
        return_value=torch.zeros((33, 1, 1, 3), dtype=torch.float32)
    )
    audio_data = np.zeros(plugin._slice_len_samples, dtype=np.float32).tobytes()
    chunk = AudioChunk(
        data=audio_data,
        sample_rate=16000,
        format="float32",
        session_id="session-1",
        question_id="question-1",
        reply_id="reply-1",
        turn_seq=1,
        user_final_unix_ms=1710000000123,
    )

    chunks = list(plugin._generate_chunks_sync(chunk))

    assert len(chunks) == 1
    assert chunks[0].trace_generation_started_since_user_final_ms >= 0


def test_config_loading():
    """Test that config can be read from cyberverse_config.yaml."""
    from inference.core.config import load_config

    config = load_config("cyberverse_config.yaml")
    assert config["inference"]["avatar"]["default"] in {"flash_head", "live_act"}
    assert set(config["inference"]["avatar"]["runtime"].keys()) == {
        "cuda_visible_devices",
        "world_size",
    }
    assert set(config["warmup"].keys()) == {"enabled", "distributed"}
    fh = config["inference"]["avatar"]["flash_head"]
    assert fh["model_type"] in {"lite", "pro", "pretrained"}
    assert fh["models_dir"] == "models"
    assert "checkpoint_dir" in fh
    assert fh["compile_model"] is True
    assert fh["compile_vae"] is True
    assert fh["dist_worker_main_thread"] is True
    assert fh["infer_params"]["frame_num"] == 33
    assert fh["infer_params"]["num_heads"] == 12
