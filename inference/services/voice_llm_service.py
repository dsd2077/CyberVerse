import asyncio
import logging

import grpc

from inference.core.registry import PluginRegistry
from inference.core.types import AudioChunk, VoiceLLMInputEvent, VoiceLLMSessionConfig
from inference.generated import common_pb2, voice_llm_pb2, voice_llm_pb2_grpc
from inference.plugins.voice_llm.base import VoiceCheckError, VoiceLLMPlugin

logger = logging.getLogger(__name__)


def _audio_chunk_to_pb(ac: AudioChunk) -> common_pb2.AudioChunk:
    return common_pb2.AudioChunk(
        data=ac.data,
        sample_rate=ac.sample_rate,
        channels=ac.channels,
        format=ac.format or "",
        is_final=ac.is_final,
        timestamp_ms=ac.timestamp_ms,
    )


def _session_config_from_pb(cfg: voice_llm_pb2.VoiceLLMConfig) -> VoiceLLMSessionConfig:
    return VoiceLLMSessionConfig(
        session_id=cfg.session_id,
        system_prompt=cfg.system_prompt,
        voice=cfg.voice,
        bot_name=cfg.bot_name,
        speaking_style=cfg.speaking_style,
        welcome_message=cfg.welcome_message,
    )


class VoiceLLMGRPCService(voice_llm_pb2_grpc.VoiceLLMServiceServicer):

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def _get_plugin(self) -> VoiceLLMPlugin:
        plugin = self.registry.get_by_category("voice_llm")
        if plugin is None:
            raise RuntimeError("No VoiceLLM plugin initialized")
        return plugin

    @staticmethod
    def _input_event_from_pb(msg: voice_llm_pb2.VoiceLLMInput) -> VoiceLLMInputEvent | None:
        which = msg.WhichOneof("input")
        if which == "audio":
            return VoiceLLMInputEvent(audio=msg.audio.data)
        if which == "text":
            return VoiceLLMInputEvent(text=msg.text)
        return None

    async def Converse(self, request_iterator, context):
        """Stream user audio/text to VoiceLLM (e.g. Doubao); yield audio + transcripts only.

        Avatar video is produced by AvatarService.GenerateStream; the Go orchestrator
        composes VoiceLLM output with that stream.
        """
        plugin = self._get_plugin()

        # Phase 1: read the config message and first input event.
        session_config: VoiceLLMSessionConfig | None = None
        first_input: VoiceLLMInputEvent | None = None
        async for msg in request_iterator:
            which = msg.WhichOneof("input")
            if which == "config":
                session_config = _session_config_from_pb(msg.config)
                logger.debug(
                    "VoiceLLM session config: voice=%r bot_name=%r system_prompt=%r welcome=%r",
                    session_config.voice,
                    session_config.bot_name,
                    session_config.system_prompt[:50] if session_config.system_prompt else "",
                    session_config.welcome_message[:50]
                    if session_config.welcome_message
                    else "",
                )
                continue
            first_input = self._input_event_from_pb(msg)
            break

        if session_config is None:
            session_config = VoiceLLMSessionConfig()
        if first_input is not None:
            session_config.input_mode = "text" if first_input.text else "keep_alive"

        # Phase 2: stream remaining messages as unified input events.
        async def input_stream():
            if first_input is not None:
                yield first_input
            async for msg in request_iterator:
                event = self._input_event_from_pb(msg)
                if event is not None:
                    yield event

        async for event in plugin.converse_stream(input_stream(), session_config=session_config):
            output = voice_llm_pb2.VoiceLLMOutput(is_final=event.is_final)
            if event.audio:
                output.audio.CopyFrom(_audio_chunk_to_pb(event.audio))
            if event.transcript:
                output.transcript = event.transcript
            if event.user_transcript:
                output.user_transcript = event.user_transcript
            if event.question_id:
                output.question_id = event.question_id
            if event.reply_id:
                output.reply_id = event.reply_id
            if event.barge_in:
                output.barge_in = True
            yield output

    async def CheckVoice(self, request, context):
        try:
            plugin = self._get_plugin()
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return voice_llm_pb2.CheckVoiceResponse(ok=False)

        try:
            await asyncio.wait_for(
                plugin.check_voice(session_config=_session_config_from_pb(request.config)),
                timeout=4.5,
            )
            return voice_llm_pb2.CheckVoiceResponse(ok=True)
        except VoiceCheckError as exc:
            return voice_llm_pb2.CheckVoiceResponse(
                ok=False,
                provider_error=str(exc),
            )
        except asyncio.TimeoutError:
            context.set_code(grpc.StatusCode.DEADLINE_EXCEEDED)
            context.set_details("voice check timed out")
            return voice_llm_pb2.CheckVoiceResponse(ok=False)
        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return voice_llm_pb2.CheckVoiceResponse(ok=False)

    async def Interrupt(self, request, context):
        plugin = self._get_plugin()
        await plugin.interrupt()
        return voice_llm_pb2.InterruptResponse(success=True)
