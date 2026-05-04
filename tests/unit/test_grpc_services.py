"""Tests for gRPC service layer using mock plugins."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from inference.core.registry import PluginRegistry
from inference.core.types import AudioChunk, PluginConfig, VideoChunk, LLMResponseChunk, TranscriptEvent
from inference.generated import asr_pb2, common_pb2, llm_pb2, tts_pb2
from inference.plugins.avatar.base import AvatarPlugin
from inference.plugins.llm.base import LLMPlugin
from inference.plugins.tts.base import TTSPlugin
from inference.plugins.asr.base import ASRPlugin
from inference.services.avatar_service import (
    AvatarGRPCService,
    FLASHHEAD_GENERATION_STARTED_HEADER,
)
from inference.services.llm_service import LLMGRPCService
from inference.services.tts_service import TTSGRPCService
from inference.services.asr_service import ASRGRPCService


# --- Mock Plugins ---

class MockAvatarPlugin(AvatarPlugin):
    name = "avatar.mock"

    async def initialize(self, config):
        pass

    async def shutdown(self):
        pass

    async def set_avatar(self, image_path, use_face_crop=False):
        self.last_image_path = image_path

    async def generate_stream(self, audio_stream):
        async for chunk in audio_stream:
            self.last_audio_chunk = chunk
            frames = np.zeros((28, 512, 512, 3), dtype=np.uint8)
            yield VideoChunk(frames=frames, fps=25, chunk_index=1, is_final=chunk.is_final)

    async def reset(self):
        pass

    def get_fps(self):
        return 25

    def get_output_dimensions(self):
        return 320, 480


class MockLLMPlugin(LLMPlugin):
    name = "llm.mock"

    async def initialize(self, config):
        pass

    async def shutdown(self):
        pass

    async def generate_stream(self, messages):
        yield LLMResponseChunk(token="Hello", accumulated_text="Hello", is_sentence_end=False)
        yield LLMResponseChunk(token="!", accumulated_text="Hello!", is_sentence_end=True, is_final=True)


class MockTTSPlugin(TTSPlugin):
    name = "tts.mock"

    async def initialize(self, config):
        pass

    async def shutdown(self):
        pass

    async def synthesize_stream(self, text_stream, request_config=None):
        self.last_request_config = request_config
        async for text in text_stream:
            self.last_text = text
            audio = np.zeros(16000, dtype=np.float32)
            yield AudioChunk(data=audio.tobytes(), sample_rate=16000, is_final=False)


class MockASRPlugin(ASRPlugin):
    name = "asr.mock"

    async def initialize(self, config):
        pass

    async def shutdown(self):
        pass

    async def transcribe_stream(self, audio_stream, request_config=None):
        self.last_request_config = request_config
        async for chunk in audio_stream:
            self.last_audio_chunk = chunk
            yield TranscriptEvent(text="hello", is_final=True, confidence=0.95)


@pytest.fixture
async def registry():
    reg = PluginRegistry()
    reg.register("avatar.mock", MockAvatarPlugin)
    reg.register("llm.mock", MockLLMPlugin)
    reg.register("tts.mock", MockTTSPlugin)
    reg.register("asr.mock", MockASRPlugin)
    await reg.initialize("avatar.mock", PluginConfig(plugin_name="avatar.mock"))
    await reg.initialize("llm.mock", PluginConfig(plugin_name="llm.mock"))
    await reg.initialize("tts.mock", PluginConfig(plugin_name="tts.mock"))
    await reg.initialize("asr.mock", PluginConfig(plugin_name="asr.mock"))
    return reg


# --- Avatar Service Tests ---

@pytest.mark.asyncio
async def test_avatar_get_info(registry):
    svc = AvatarGRPCService(registry)
    request = MagicMock()
    context = MagicMock()
    info = await svc.GetInfo(request, context)
    assert info.model_name == "avatar.mock"
    assert info.output_fps == 25
    assert info.output_width == 320
    assert info.output_height == 480


@pytest.mark.asyncio
async def test_avatar_reset(registry):
    svc = AvatarGRPCService(registry)
    request = MagicMock(session_id="test")
    context = MagicMock()
    resp = await svc.Reset(request, context)
    assert resp.success is True


@pytest.mark.asyncio
async def test_avatar_set_avatar(registry):
    svc = AvatarGRPCService(registry)
    request = MagicMock(image_data=b"fake_png", image_format="png", use_face_crop=False)
    context = MagicMock()
    resp = await svc.SetAvatar(request, context)
    assert resp.success is True


@pytest.mark.asyncio
async def test_avatar_generate_stream(registry):
    svc = AvatarGRPCService(registry)
    context = MagicMock()
    context.send_initial_metadata = AsyncMock()
    context.invocation_metadata.return_value = (
        ("x-cyberverse-session-id", "session-1"),
        ("x-cyberverse-question-id", "question-1"),
        ("x-cyberverse-reply-id", "reply-1"),
        ("x-cyberverse-turn-seq", "7"),
        ("x-cyberverse-user-final-unix-ms", "1710000000123"),
    )

    audio_data = np.zeros(17920, dtype=np.float32).tobytes()

    async def request_iterator():
        chunk = MagicMock()
        chunk.data = audio_data
        chunk.sample_rate = 16000
        chunk.channels = 1
        chunk.format = "float32"
        chunk.is_final = True
        chunk.timestamp_ms = 0
        yield chunk

    results = []
    async for vc in svc.GenerateStream(request_iterator(), context):
        results.append(vc)

    assert len(results) == 1
    assert results[0].num_frames == 28
    assert results[0].fps == 25
    plugin = registry.get("avatar.mock")
    assert plugin.last_audio_chunk.session_id == "session-1"
    assert plugin.last_audio_chunk.question_id == "question-1"
    assert plugin.last_audio_chunk.reply_id == "reply-1"
    assert plugin.last_audio_chunk.turn_seq == 7
    assert plugin.last_audio_chunk.user_final_unix_ms == 1710000000123
    context.send_initial_metadata.assert_not_awaited()


@pytest.mark.asyncio
async def test_avatar_generate_stream_sends_flashhead_trace_metadata(registry):
    svc = AvatarGRPCService(registry)
    plugin = registry.get("avatar.mock")

    async def generate_stream(audio_stream):
        async for chunk in audio_stream:
            plugin.last_audio_chunk = chunk
            frames = np.zeros((28, 512, 512, 3), dtype=np.uint8)
            yield VideoChunk(
                frames=frames,
                fps=25,
                chunk_index=1,
                is_final=chunk.is_final,
                trace_generation_started_since_user_final_ms=389,
            )

    plugin.generate_stream = generate_stream

    context = MagicMock()
    context.send_initial_metadata = AsyncMock()
    context.invocation_metadata.return_value = (
        ("x-cyberverse-session-id", "session-1"),
        ("x-cyberverse-question-id", "question-1"),
        ("x-cyberverse-reply-id", "reply-1"),
        ("x-cyberverse-turn-seq", "7"),
        ("x-cyberverse-user-final-unix-ms", "1710000000123"),
    )

    async def request_iterator():
        chunk = MagicMock()
        chunk.data = np.zeros(17920, dtype=np.float32).tobytes()
        chunk.sample_rate = 16000
        chunk.channels = 1
        chunk.format = "float32"
        chunk.is_final = True
        chunk.timestamp_ms = 0
        yield chunk

    results = []
    async for vc in svc.GenerateStream(request_iterator(), context):
        results.append(vc)

    assert len(results) == 1
    context.send_initial_metadata.assert_awaited_once_with(
        ((FLASHHEAD_GENERATION_STARTED_HEADER, "389"),)
    )


# --- LLM Service Tests ---

@pytest.mark.asyncio
async def test_llm_generate_stream(registry):
    svc = LLMGRPCService(registry)
    request = llm_pb2.LLMRequest(
        session_id="session-1",
        messages=[llm_pb2.ChatMessage(role="user", content="Hi")],
        config=llm_pb2.LLMConfig(provider="mock"),
    )
    context = MagicMock()

    results = []
    async for chunk in svc.GenerateStream(request, context):
        results.append(chunk)

    assert len(results) == 2
    assert results[0].token == "Hello"
    assert results[1].is_final is True


@pytest.mark.asyncio
async def test_llm_rejects_images_when_plugin_does_not_support_them(registry):
    svc = LLMGRPCService(registry)
    request = llm_pb2.LLMRequest(
        session_id="session-1",
        messages=[
            llm_pb2.ChatMessage(
                role="user",
                content="What is on screen?",
                images=[
                    common_pb2.ImageFrame(
                        data=b"\xff\xd8\xff",
                        mime_type="image/jpeg",
                        width=1,
                        height=1,
                        source="screen",
                        timestamp_ms=123,
                        frame_seq=1,
                    )
                ],
            )
        ],
        config=llm_pb2.LLMConfig(provider="mock"),
    )

    with pytest.raises(RuntimeError, match="does not support image input"):
        async for _ in svc.GenerateStream(request, MagicMock()):
            pass


# --- TTS Service Tests ---

@pytest.mark.asyncio
async def test_tts_synthesize_stream(registry):
    svc = TTSGRPCService(registry)
    context = MagicMock()

    async def text_stream():
        yield tts_pb2.TextChunk(
            config=tts_pb2.TTSConfig(
                provider="mock",
                voice="nova",
                session_id="session-1",
            )
        )
        yield tts_pb2.TextChunk(text="Hello world")

    results = []
    async for ac in svc.SynthesizeStream(text_stream(), context):
        results.append(ac)

    assert len(results) == 1
    assert results[0].sample_rate == 16000
    plugin = registry.get("tts.mock")
    assert plugin.last_text == "Hello world"
    assert plugin.last_request_config.provider == "mock"
    assert plugin.last_request_config.voice == "nova"
    assert plugin.last_request_config.session_id == "session-1"


# --- ASR Service Tests ---

@pytest.mark.asyncio
async def test_asr_transcribe_stream(registry):
    svc = ASRGRPCService(registry)
    context = MagicMock()

    async def audio_stream():
        yield asr_pb2.ASRInput(
            config=asr_pb2.ASRConfig(
                provider="mock",
                language="zh",
                session_id="session-1",
            )
        )
        yield asr_pb2.ASRInput(
            audio=common_pb2.AudioChunk(
                data=b"\x00" * 3200,
                sample_rate=16000,
                channels=1,
                format="pcm_s16le",
            )
        )

    results = []
    async for event in svc.TranscribeStream(audio_stream(), context):
        results.append(event)

    assert len(results) == 1
    assert results[0].text == "hello"
    assert results[0].is_final is True
    plugin = registry.get("asr.mock")
    assert plugin.last_audio_chunk == b"\x00" * 3200
    assert plugin.last_request_config.provider == "mock"
    assert plugin.last_request_config.language == "zh"
    assert plugin.last_request_config.session_id == "session-1"
