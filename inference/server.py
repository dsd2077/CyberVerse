"""CyberVerse gRPC Inference Server entry point."""
import argparse
import asyncio
import logging
import os
import signal
import warnings

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from inference.core.config import load_config
from inference.core.registry import PluginRegistry, import_plugin_class
from inference.core.types import PluginConfig
from inference.generated import (
    avatar_pb2_grpc,
    llm_pb2_grpc,
    tts_pb2_grpc,
    asr_pb2_grpc,
    voice_llm_pb2_grpc,
)
from inference.services.avatar_service import AvatarGRPCService
from inference.services.llm_service import LLMGRPCService
from inference.services.tts_service import TTSGRPCService
from inference.services.asr_service import ASRGRPCService
from inference.services.voice_llm_service import VoiceLLMGRPCService

logger = logging.getLogger(__name__)

# SageAttention calls PyCapsule CUDA helpers; torch.compile/dynamo emits verbose UserWarnings
# though execution still falls back correctly. Suppress only this known noise at process start.
warnings.filterwarnings(
    "ignore",
    message=r".*Dynamo does not know how to trace the builtin `sageattention\._fused\..*",
    category=UserWarning,
)

_PLUGIN_CATEGORIES = ("avatar", "llm", "tts", "asr", "voice_llm")
_INITIALIZE_ALL_CATEGORIES = {"llm", "tts", "asr", "voice_llm"}


def _configure_process_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    # LiveAct pulls in vLLM transitively but this server path does not use it.
    # Keep real errors while dropping its startup noise.
    logging.getLogger("vllm").setLevel(logging.ERROR)


class InferenceServer:
    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)
        self.registry = PluginRegistry()
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_primary = self.world_size <= 1 or self.rank == 0
        self._worker_stop = asyncio.Event()
        self._stop_lock = asyncio.Lock()
        self._stopped = False
        self.server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", 50 * 1024 * 1024),
                ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.min_ping_interval_without_data_ms", 30000),
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 30000),
            ]
        )

    def _build_plugin_config(
        self, category: str, full_name: str, conf: dict
    ) -> PluginConfig:
        """Build plugin config with per-plugin params and shared root settings."""
        params = {k: v for k, v in conf.items() if k != "plugin_class"}
        shared: dict[str, object] = {}
        if category == "avatar":
            avatar = self.config.get("inference", {}).get("avatar", {})
            runtime = avatar.get("runtime")
            if isinstance(runtime, dict):
                params = {**runtime, **params}
            warmup = self.config.get("warmup")
            if isinstance(warmup, dict):
                shared["warmup"] = warmup
        return PluginConfig(
            plugin_name=full_name,
            params=params,
            shared=shared,
        )

    def _register_plugins(self) -> None:
        """Discover and register plugin classes from config (no hardcoded imports)."""
        for category in _PLUGIN_CATEGORIES:
            section = self.config.get("inference", {}).get(category, {})
            for name, conf in section.items():
                if name == "default" or not isinstance(conf, dict):
                    continue
                class_path = conf.get("plugin_class")
                if not class_path:
                    if self.is_primary:
                        logger.debug("No plugin_class for %s.%s, skipping", category, name)
                    continue
                full_name = f"{category}.{name}"
                try:
                    cls = import_plugin_class(class_path)
                    self.registry.register(full_name, cls)
                    if self.is_primary:
                        logger.info("Registered plugin: %s -> %s", full_name, class_path)
                except (ImportError, AttributeError, TypeError) as e:
                    if self.is_primary:
                        logger.warning("Plugin %s not available: %s", full_name, e)

    async def _initialize_configured_plugins(self) -> None:
        """Initialize configured plugins.

        LLM/ASR/TTS/VoiceLLM are lightweight component plugins and can be
        selected per request, so initialize every configured entry. Avatar
        stays default-only to avoid extra model/GPU cost.
        """
        for category in _PLUGIN_CATEGORIES:
            section = self.config.get("inference", {}).get(category, {})
            if category in _INITIALIZE_ALL_CATEGORIES:
                names = [
                    name
                    for name, conf in section.items()
                    if name != "default" and isinstance(conf, dict)
                ]
            else:
                default_name = section.get("default")
                names = [default_name] if default_name else []

            for name in names:
                full_name = f"{category}.{name}"
                if full_name not in self.registry.registered_names:
                    continue
                conf = section.get(name, {})
                plugin_config = self._build_plugin_config(category, full_name, conf)
                try:
                    await self.registry.initialize(full_name, plugin_config)
                    if self.is_primary:
                        logger.info("Initialized plugin: %s", full_name)
                    if category == "avatar" and self.is_primary:
                        logger.info("Active avatar model initialized: %s", name)
                except Exception:
                    logger.exception("Failed to initialize plugin: %s", full_name)

    def _register_grpc_services(self) -> None:
        avatar_pb2_grpc.add_AvatarServiceServicer_to_server(
            AvatarGRPCService(self.registry), self.server
        )
        llm_pb2_grpc.add_LLMServiceServicer_to_server(
            LLMGRPCService(self.registry), self.server
        )
        tts_pb2_grpc.add_TTSServiceServicer_to_server(
            TTSGRPCService(self.registry), self.server
        )
        asr_pb2_grpc.add_ASRServiceServicer_to_server(
            ASRGRPCService(self.registry), self.server
        )
        voice_llm_pb2_grpc.add_VoiceLLMServiceServicer_to_server(
            VoiceLLMGRPCService(self.registry), self.server
        )

        health_servicer = health.HealthServicer()
        health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self.server)

    async def start(self) -> None:
        self._register_plugins()
        self._register_grpc_services()
        await self._initialize_configured_plugins()

        # torchrun multi-process mode: only rank0 binds gRPC; other ranks stay
        # alive as distributed workers for FlashHead model parallel inference.
        if self.world_size > 1 and self.rank != 0:
            logger.info(
                "Inference worker rank started: rank=%d/%d (gRPC disabled, waiting for shutdown)",
                self.rank,
                self.world_size,
            )
            await self._worker_stop.wait()
            return

        port = self.config.get("server", {}).get("grpc_port", 50051)
        self.server.add_insecure_port(f"[::]:{port}")
        await self.server.start()
        logger.info("CyberVerse Inference Server started on port %d", port)
        logger.info("Registered plugins: %s", self.registry.registered_names)
        logger.info("Initialized plugins: %s", self.registry.initialized_names)
        await self.server.wait_for_termination()

    async def stop(self) -> None:
        async with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True

        logger.info("Inference server stopping (rank=%d)...", self.rank)
        await self.registry.shutdown_all()
        if self.world_size > 1 and self.rank != 0:
            self._worker_stop.set()
            return
        await self.server.stop(grace=5)


async def main(config_path: str) -> None:
    _configure_process_logging()
    server = InferenceServer(config_path)

    loop = asyncio.get_running_loop()

    def _on_signal() -> None:
        # Avoid duplicate tasks if the user hits Ctrl+C repeatedly.
        asyncio.create_task(server.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    try:
        await server.start()
    except Exception:
        logger.exception("Server error")
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CyberVerse Inference Server")
    parser.add_argument("--config", default="cyberverse_config.yaml")
    args = parser.parse_args()
    asyncio.run(main(args.config))
