import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from inference.core.types import PluginConfig, VoiceLLMInputEvent, VoiceLLMSessionConfig
from inference.plugins.voice_llm.qwen_omni_realtime import QwenOmniRealtimePlugin


class FakeQwenWS:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.closed = False

    async def send(self, payload: str):
        self.sent.append(json.loads(payload))

    async def recv(self):
        if not self.events:
            raise RuntimeError("no fake websocket events left")
        return json.dumps(self.events.pop(0), ensure_ascii=False)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return json.dumps(self.events.pop(0), ensure_ascii=False)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_initialize_sets_qwen_omni_defaults():
    plugin = QwenOmniRealtimePlugin()

    await plugin.initialize(
        PluginConfig(
            plugin_name="voice_llm.qwen_omni",
            params={
                "api_key": "dashscope-key",
                "model": "qwen3.5-omni-flash-realtime",
                "voice": "Tina",
            },
        )
    )

    assert plugin.api_key == "dashscope-key"
    assert plugin.model == "qwen3.5-omni-flash-realtime"
    assert plugin.voice == "Tina"
    assert plugin.ws_url.endswith("model=qwen3.5-omni-flash-realtime")


@pytest.mark.asyncio
async def test_check_voice_configures_session_with_voice_override():
    plugin = QwenOmniRealtimePlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="voice_llm.qwen_omni",
            params={"api_key": "dashscope-key"},
        )
    )
    ws = FakeQwenWS([{"type": "session.updated"}])
    websockets = SimpleNamespace(connect=AsyncMock(return_value=ws))

    with patch.dict("sys.modules", {"websockets": websockets}):
        await plugin.check_voice(VoiceLLMSessionConfig(voice="Cindy"))

    assert ws.closed is True
    assert ws.sent[0]["type"] == "session.update"
    session = ws.sent[0]["session"]
    assert session["voice"] == "Cindy"
    assert session["modalities"] == ["text", "audio"]
    assert session["input_audio_format"] == "pcm"
    assert session["output_audio_format"] == "pcm"
    assert session["turn_detection"]["type"] == "semantic_vad"


@pytest.mark.asyncio
async def test_converse_stream_emits_audio_transcripts_and_final():
    plugin = QwenOmniRealtimePlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="voice_llm.qwen_omni",
            params={"api_key": "dashscope-key"},
        )
    )
    audio_bytes = b"\x01\x00\x02\x00"
    ws = FakeQwenWS(
        [
            {"type": "session.created"},
            {"type": "input_audio_buffer.speech_started"},
            {
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": "你好",
            },
            {"type": "response.audio_transcript.delta", "delta": "收到"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(audio_bytes).decode("ascii"),
            },
            {"type": "response.audio_transcript.done", "transcript": "收到"},
            {"type": "response.done"},
        ]
    )
    websockets = SimpleNamespace(connect=AsyncMock(return_value=ws))

    async def inputs():
        yield VoiceLLMInputEvent(audio=b"\x03\x00")

    with patch.dict("sys.modules", {"websockets": websockets}):
        outputs = [
            event
            async for event in plugin.converse_stream(
                inputs(),
                VoiceLLMSessionConfig(session_id="session-1", voice="Tina"),
            )
        ]

    sent_audio = [event for event in ws.sent if event["type"] == "input_audio_buffer.append"]
    assert sent_audio
    assert base64.b64decode(sent_audio[0]["audio"]) == b"\x03\x00"

    assert outputs[0].barge_in is True
    assert outputs[1].user_transcript == "你好"
    assert outputs[2].transcript == "收到"
    assert outputs[3].audio is not None
    assert outputs[3].audio.data == audio_bytes
    assert outputs[3].audio.sample_rate == 24000
    assert outputs[4].is_final is True
    assert outputs[4].transcript == "收到"
    assert outputs[4].audio is not None
    assert outputs[4].audio.is_final is True
