"""Tests for gRPC service layer using mock plugins."""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from inference.core.registry import PluginRegistry
from inference.core.types import AudioChunk, PluginConfig, VideoChunk, LLMResponseChunk, TranscriptEvent
from inference.plugins.avatar.base import AvatarPlugin
from inference.plugins.llm.base import LLMPlugin
from inference.plugins.tts.base import TTSPlugin
from inference.plugins.asr.base import ASRPlugin
from inference.services.avatar_service import AvatarGRPCService
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

    async def synthesize_stream(self, text_stream):
        async for text in text_stream:
            audio = np.zeros(16000, dtype=np.float32)
            yield AudioChunk(data=audio.tobytes(), sample_rate=16000, is_final=False)


class MockASRPlugin(ASRPlugin):
    name = "asr.mock"

    async def initialize(self, config):
        pass

    async def shutdown(self):
        pass

    async def transcribe_stream(self, audio_stream):
        async for _ in audio_stream:
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
    context.invocation_metadata.return_value = (
        ("x-cyberverse-session-id", "session-1"),
        ("x-cyberverse-question-id", "question-1"),
        ("x-cyberverse-reply-id", "reply-1"),
        ("x-cyberverse-turn-seq", "7"),
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


# --- LLM Service Tests ---

@pytest.mark.asyncio
async def test_llm_generate_stream(registry):
    svc = LLMGRPCService(registry)
    request = MagicMock()
    msg = MagicMock()
    msg.role = "user"
    msg.content = "Hi"
    request.messages = [msg]
    context = MagicMock()

    results = []
    async for chunk in svc.GenerateStream(request, context):
        results.append(chunk)

    assert len(results) == 2
    assert results[0].token == "Hello"
    assert results[1].is_final is True


# --- TTS Service Tests ---

@pytest.mark.asyncio
async def test_tts_synthesize_stream(registry):
    svc = TTSGRPCService(registry)
    context = MagicMock()

    async def text_stream():
        chunk = MagicMock()
        chunk.text = "Hello world"
        yield chunk

    results = []
    async for ac in svc.SynthesizeStream(text_stream(), context):
        results.append(ac)

    assert len(results) == 1
    assert results[0].sample_rate == 16000


# --- ASR Service Tests ---

@pytest.mark.asyncio
async def test_asr_transcribe_stream(registry):
    svc = ASRGRPCService(registry)
    context = MagicMock()

    async def audio_stream():
        chunk = MagicMock()
        chunk.data = b"\x00" * 3200
        yield chunk

    results = []
    async for event in svc.TranscribeStream(audio_stream(), context):
        results.append(event)

    assert len(results) == 1
    assert results[0].text == "hello"
    assert results[0].is_final is True
