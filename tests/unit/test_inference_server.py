from unittest.mock import MagicMock, patch

import pytest

from inference.server import InferenceServer


def _make_server(config: dict) -> InferenceServer:
    with patch("inference.server.load_config", return_value=config):
        with patch("inference.server.grpc.aio.server", return_value=MagicMock()):
            return InferenceServer("cyberverse_config.yaml")


def test_build_plugin_config_passes_root_warmup_to_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.flash_head",
        {
            "plugin_class": "pkg.Plugin",
            "device": "cuda:0",
            "compile_model": True,
            "compile_vae": True,
            "dist_worker_main_thread": True,
            "infer_params": {
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
            },
        },
    )

    assert plugin_config.plugin_name == "avatar.flash_head"
    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 2,
        "device": "cuda:0",
        "compile_model": True,
        "compile_vae": True,
        "dist_worker_main_thread": True,
        "infer_params": {
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
        },
    }
    assert plugin_config.shared["warmup"] == config["warmup"]


def test_build_plugin_config_model_values_override_avatar_runtime_defaults():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "avatar",
        "avatar.live_act",
        {
            "plugin_class": "pkg.Plugin",
            "world_size": 1,
            "compile_wan_model": False,
            "compile_vae_decode": False,
            "dist_worker_main_thread": True,
            "infer_params": {
                "size": "320*480",
                "fps": 20,
                "audio_cfg": 1.0,
            },
        },
    )

    assert plugin_config.params == {
        "cuda_visible_devices": "0,1",
        "world_size": 1,
        "compile_wan_model": False,
        "compile_vae_decode": False,
        "dist_worker_main_thread": True,
        "infer_params": {
            "size": "320*480",
            "fps": 20,
            "audio_cfg": 1.0,
        },
    }


def test_build_plugin_config_does_not_pass_root_warmup_to_non_avatar_plugins():
    config = {
        "inference": {
            "avatar": {
                "runtime": {
                    "cuda_visible_devices": "0,1",
                    "world_size": 2,
                }
            }
        },
        "warmup": {
            "enabled": True,
            "distributed": {"enabled": True, "timeout_s": 30},
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "llm",
        "llm.openai",
        {
            "plugin_class": "pkg.Plugin",
            "model": "gpt-4o",
        },
    )

    assert plugin_config.plugin_name == "llm.openai"
    assert plugin_config.params == {"model": "gpt-4o"}
    assert plugin_config.shared == {}


def test_build_plugin_config_passes_omni_models_to_persona_plugins():
    config = {
        "inference": {
            "omni": {
                "qwen_omni": {
                    "plugin_class": "pkg.Qwen",
                    "model": "qwen3.5-omni-flash-realtime",
                }
            },
            "persona": {
                "persona": {
                    "plugin_class": "pkg.Persona",
                    "model_provider": "qwen_omni",
                }
            },
        }
    }
    server = _make_server(config)

    plugin_config = server._build_plugin_config(
        "persona",
        "persona.persona",
        config["inference"]["persona"]["persona"],
    )

    assert plugin_config.plugin_name == "persona.persona"
    assert plugin_config.params == {"model_provider": "qwen_omni"}
    assert plugin_config.shared["omni"] == config["inference"]["omni"]
    assert plugin_config.shared["runtime_config"] == config


def test_register_plugins_skips_avatar_when_disabled_but_keeps_voice_plugins():
    config = {
        "inference": {
            "avatar": {
                "enabled": False,
                "default": "flash_head",
                "flash_head": {"plugin_class": "pkg.Avatar"},
            },
            "omni": {
                "default": "qwen_omni",
                "qwen_omni": {"plugin_class": "pkg.Omni"},
            },
        }
    }
    server = _make_server(config)

    with patch("inference.server.import_plugin_class", return_value=object):
        server._register_plugins()

    assert "avatar.flash_head" not in server.registry.registered_names
    assert "omni.qwen_omni" in server.registry.registered_names


@pytest.mark.asyncio
async def test_initialize_configured_plugins_skips_avatar_when_disabled():
    class DummyPlugin:
        async def initialize(self, config):
            self.config = config

    config = {
        "inference": {
            "avatar": {
                "enabled": False,
                "default": "flash_head",
                "flash_head": {"plugin_class": "pkg.Avatar"},
            },
            "tts": {
                "default": "qwen",
                "qwen": {"plugin_class": "pkg.TTS"},
            },
        }
    }
    server = _make_server(config)
    server.registry.register("avatar.flash_head", DummyPlugin)
    server.registry.register("tts.qwen", DummyPlugin)

    await server._initialize_configured_plugins()

    assert "avatar.flash_head" not in server.registry.initialized_names
    assert "tts.qwen" in server.registry.initialized_names
