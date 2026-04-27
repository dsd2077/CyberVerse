from unittest.mock import patch

import pytest

pytest.importorskip("torch")

from inference.core.types import PluginConfig
from inference.plugins.avatar.live_act_plugin import LiveActAvatarPlugin


def _config(*, warmup_enabled: bool) -> PluginConfig:
    return PluginConfig(
        plugin_name="avatar.live_act",
        params={
            "world_size": 1,
            "seed": 42,
            "t5_cpu": True,
            "fp8_kv_cache": False,
            "offload_cache": False,
            "block_offload": False,
            "mean_memory": False,
            "dist_worker_main_thread": True,
            "default_prompt": "一个人在说话",
            "ckpt_dir": "/tmp/liveact",
            "wav2vec_dir": "/tmp/wav2vec",
            "infer_params": {
                "size": "320*480",
                "fps": 20,
                "audio_cfg": 1.0,
            },
        },
        shared={
            "warmup": {
                "enabled": warmup_enabled,
                "distributed": {"enabled": True, "timeout_s": 30},
            }
        },
    )


def test_init_sync_runs_warmup_after_avatar_setup():
    plugin = LiveActAvatarPlugin()
    order: list[str] = []

    with patch.object(plugin, "_load_models"):
        with patch.object(plugin, "_init_kv_cache"):
            with patch.object(
                plugin,
                "_create_default_avatar_placeholder",
                return_value=("/tmp/avatar.png", False),
            ):
                with patch.object(
                    plugin,
                    "_set_avatar_sync_local",
                    side_effect=lambda image_path: (
                        order.append("avatar"),
                        setattr(plugin, "_avatar_initialized", True),
                    ),
                ):
                    with patch.object(
                        plugin,
                        "_warmup",
                        side_effect=lambda: order.append("warmup"),
                    ) as warmup:
                        plugin._init_sync(_config(warmup_enabled=True))

    warmup.assert_called_once_with()
    assert order == ["avatar", "warmup"]
    assert plugin._width == 320
    assert plugin._height == 480
    assert plugin._fps == 20
    assert plugin._audio_cfg == 1.0


def test_init_sync_skips_warmup_when_disabled():
    plugin = LiveActAvatarPlugin()
    order: list[str] = []

    with patch.object(plugin, "_load_models"):
        with patch.object(plugin, "_init_kv_cache"):
            with patch.object(
                plugin,
                "_create_default_avatar_placeholder",
                return_value=("/tmp/avatar.png", False),
            ):
                with patch.object(
                    plugin,
                    "_set_avatar_sync_local",
                    side_effect=lambda image_path: (
                        order.append("avatar"),
                        setattr(plugin, "_avatar_initialized", True),
                    ),
                ):
                    with patch.object(plugin, "_warmup") as warmup:
                        plugin._init_sync(_config(warmup_enabled=False))

    warmup.assert_not_called()
    assert order == ["avatar"]


def test_get_output_dimensions_aligns_to_vae_stride():
    plugin = LiveActAvatarPlugin()
    plugin._width = 256
    plugin._height = 417

    assert plugin.get_output_dimensions() == (256, 416)
