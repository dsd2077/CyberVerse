import logging

import grpc

from inference.core.types import ASRRequestConfig
from inference.core.registry import PluginRegistry
from inference.generated import asr_pb2, asr_pb2_grpc
from inference.plugins.asr.base import ASRPlugin

logger = logging.getLogger(__name__)


class ASRGRPCService(asr_pb2_grpc.ASRServiceServicer):

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def _get_plugin(self, provider: str = "") -> ASRPlugin:
        provider = provider.strip()
        if provider:
            return self.registry.get(f"asr.{provider}")
        plugin = self.registry.get_by_category("asr")
        if plugin is None:
            raise RuntimeError("No ASR plugin initialized")
        return plugin

    async def TranscribeStream(self, request_iterator, context):
        iterator = request_iterator.__aiter__()
        try:
            first = await anext(iterator)
        except StopAsyncIteration:
            return

        request_config = ASRRequestConfig()
        first_audio = None
        input_type = first.WhichOneof("input")
        if input_type == "config":
            request_config = ASRRequestConfig(
                provider=first.config.provider,
                language=first.config.language,
                session_id=first.config.session_id,
            )
        elif input_type == "audio":
            first_audio = first.audio.data

        try:
            plugin = self._get_plugin(request_config.provider)
        except (KeyError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        async def audio_stream():
            if first_audio:
                yield first_audio
            async for chunk in iterator:
                if chunk.WhichOneof("input") == "audio":
                    yield chunk.audio.data

        try:
            async for event in plugin.transcribe_stream(audio_stream(), request_config):
                yield asr_pb2.TranscriptEvent(
                    text=event.text,
                    is_final=event.is_final,
                    language=event.language,
                    confidence=event.confidence,
                )
        except RuntimeError as exc:
            logger.warning("ASR plugin failed: %s", exc)
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
