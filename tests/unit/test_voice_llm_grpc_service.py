"""Tests for VoiceLLM gRPC service (audio-only; avatar is AvatarService)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from inference.core.types import AudioChunk, VoiceLLMOutputEvent
from inference.services.voice_llm_service import VoiceLLMGRPCService
from inference.plugins.voice_llm.base import VoiceCheckError


@pytest.mark.asyncio
async def test_converse_yields_audio_only():
    reg = MagicMock()
    voice = MagicMock()

    async def fake_converse(_stream, session_config=None):
        yield VoiceLLMOutputEvent(
            audio=AudioChunk(
                data=b"\x01\x00",
                sample_rate=24000,
                format="pcm_s16le",
                is_final=False,
            ),
        )

    voice.converse_stream = fake_converse
    reg.get_by_category = MagicMock(return_value=voice)

    svc = VoiceLLMGRPCService(reg)

    from inference.generated import common_pb2, voice_llm_pb2

    async def requests():
        yield voice_llm_pb2.VoiceLLMInput(
            audio=common_pb2.AudioChunk(data=b"pcm", sample_rate=16000)
        )

    ctx = MagicMock()
    outs = []
    async for o in svc.Converse(requests(), ctx):
        outs.append(o)
    assert len(outs) == 1
    assert outs[0].audio.data == b"\x01\x00"
    assert outs[0].audio.sample_rate == 24000


@pytest.mark.asyncio
async def test_converse_copies_turn_metadata():
    reg = MagicMock()
    voice = MagicMock()

    async def fake_converse(_stream, session_config=None):
        yield VoiceLLMOutputEvent(
            transcript="你好",
            user_transcript="打断一下",
            is_final=True,
            question_id="question-1",
            reply_id="reply-1",
            barge_in=True,
        )

    voice.converse_stream = fake_converse
    reg.get_by_category = MagicMock(return_value=voice)

    svc = VoiceLLMGRPCService(reg)

    from inference.generated import voice_llm_pb2

    async def requests():
        yield voice_llm_pb2.VoiceLLMInput(text="hello")

    outs = []
    async for o in svc.Converse(requests(), MagicMock()):
        outs.append(o)

    assert len(outs) == 1
    assert outs[0].transcript == "你好"
    assert outs[0].user_transcript == "打断一下"
    assert outs[0].question_id == "question-1"
    assert outs[0].reply_id == "reply-1"
    assert outs[0].barge_in is True


@pytest.mark.asyncio
async def test_converse_without_voice_llm_plugin_raises():
    reg = MagicMock()
    reg.get_by_category = MagicMock(return_value=None)
    svc = VoiceLLMGRPCService(reg)

    from inference.generated import common_pb2, voice_llm_pb2

    async def requests():
        yield voice_llm_pb2.VoiceLLMInput(
            audio=common_pb2.AudioChunk(data=b"x", sample_rate=16000)
        )

    with pytest.raises(RuntimeError, match="No VoiceLLM"):
        async for _ in svc.Converse(requests(), MagicMock()):
            pass


@pytest.mark.asyncio
async def test_check_voice_returns_ok():
    reg = MagicMock()
    voice = MagicMock()
    voice.check_voice = AsyncMock(return_value=None)
    reg.get_by_category = MagicMock(return_value=voice)

    svc = VoiceLLMGRPCService(reg)

    from inference.generated import voice_llm_pb2

    req = voice_llm_pb2.CheckVoiceRequest(
        config=voice_llm_pb2.VoiceLLMConfig(voice="温柔文雅")
    )
    ctx = MagicMock()

    resp = await svc.CheckVoice(req, ctx)

    assert resp.ok is True
    assert resp.provider_error == ""
    ctx.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_check_voice_returns_provider_error():
    reg = MagicMock()
    voice = MagicMock()

    async def fake_check_voice(session_config=None):
        raise VoiceCheckError("raw provider error")

    voice.check_voice = fake_check_voice
    reg.get_by_category = MagicMock(return_value=voice)

    svc = VoiceLLMGRPCService(reg)

    from inference.generated import voice_llm_pb2

    req = voice_llm_pb2.CheckVoiceRequest(
        config=voice_llm_pb2.VoiceLLMConfig(voice="S_123456")
    )
    ctx = MagicMock()

    resp = await svc.CheckVoice(req, ctx)

    assert resp.ok is False
    assert resp.provider_error == "raw provider error"
    ctx.set_code.assert_not_called()


@pytest.mark.asyncio
async def test_check_voice_uses_provider_specific_plugin():
    reg = MagicMock()
    voice = MagicMock()
    voice.check_voice = AsyncMock(return_value=None)
    reg.get = MagicMock(return_value=voice)
    reg.get_by_category = MagicMock()

    svc = VoiceLLMGRPCService(reg)

    from inference.generated import voice_llm_pb2

    req = voice_llm_pb2.CheckVoiceRequest(
        config=voice_llm_pb2.VoiceLLMConfig(
            provider="qwen_omni",
            voice="Tina",
        )
    )
    ctx = MagicMock()

    resp = await svc.CheckVoice(req, ctx)

    assert resp.ok is True
    reg.get.assert_called_once_with("voice_llm.qwen_omni")
    reg.get_by_category.assert_not_called()
    passed_config = voice.check_voice.await_args.kwargs["session_config"]
    assert passed_config.provider == "qwen_omni"
    assert passed_config.voice == "Tina"
