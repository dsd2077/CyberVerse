"""Tests for Doubao VoiceLLM plugin using mocked WebSocket (binary protocol)."""

import json
import uuid
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference.core.types import (
    PluginConfig,
    VoiceLLMDialogContextItem,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.doubao_realtime import DoubaoRealtimePlugin
from inference.plugins.voice_llm.doubao_config import DoubaoSessionConfig, SC20_VOICES
from inference.plugins.voice_llm.doubao_protocol import (
    COMPRESSION_GZIP,
    compress_payload,
    decode_frame,
    decompress_payload,
    encode_frame,
    DoubaoEvent,
    MSGTYPE_AUDIO_ONLY_SERVER,
    MSGTYPE_ERROR,
    MSGTYPE_FULL_SERVER,
    MSGTYPE_FULL_CLIENT,
    SERIALIZATION_JSON,
    SERIALIZATION_RAW,
)


class TestDoubaoRealtimePlugin:
    @staticmethod
    def _error_frame(error_code: int, payload: bytes) -> bytes:
        frame = bytearray([0x11, MSGTYPE_ERROR | 0x04, SERIALIZATION_JSON | COMPRESSION_GZIP, 0x00])
        frame.extend(error_code.to_bytes(4, "big"))
        frame.extend(int(DoubaoEvent.START_SESSION).to_bytes(4, "big", signed=True))
        frame.extend((0).to_bytes(4, "big"))
        frame.extend(len(payload).to_bytes(4, "big"))
        frame.extend(payload)
        return bytes(frame)

    def test_name(self):
        assert DoubaoRealtimePlugin.name == "voice_llm.doubao"

    @pytest.mark.asyncio
    async def test_initialize(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "test_key",
                "app_id": "test_app",
                "ws_url": "wss://test.example.com",
                "voice_type": "zh_male_default",
                "system_prompt": "Be helpful.",
            },
        )
        await plugin.initialize(config)
        assert plugin._config.access_token == "test_key"
        assert plugin._config.app_id == "test_app"
        assert plugin._config.voice_type == "zh_male_default"
        assert plugin._config.system_prompt == "Be helpful."

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={"ws_url": "wss://test.example.com"},
        )
        with pytest.raises(ValueError, match="access_token"):
            await plugin.initialize(config)

    @pytest.mark.asyncio
    async def test_initialize_missing_ws_url_uses_default(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={"access_token": "test_key"},
        )
        await plugin.initialize(config)
        assert plugin._config.ws_url == "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"

    @pytest.mark.asyncio
    async def test_initialize_empty_ws_url_raises(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={"access_token": "test_key", "ws_url": ""},
        )
        with pytest.raises(ValueError, match="ws_url"):
            await plugin.initialize(config)

    def test_encode_decode_roundtrip(self):
        session_id = "test-session"
        payload = b"abc"
        frame = encode_frame(
            msg_type_bits=MSGTYPE_FULL_CLIENT,
            serialization_bits=SERIALIZATION_JSON,
            event=100,
            session_id=session_id,
            payload=payload,
        )
        decoded = decode_frame(frame)
        assert decoded.msg_type_bits == MSGTYPE_FULL_CLIENT
        assert decoded.event == 100
        assert decoded.session_id == session_id
        assert decoded.payload == payload

    @pytest.mark.asyncio
    async def test_converse_stream_with_mock_ws(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "zh_female_default",
                "system_prompt": "Hello",
            },
        )
        await plugin.initialize(config)

        session_uuid = uuid.UUID("00000000-0000-0000-0000-000000000001")
        connect_uuid = uuid.UUID("00000000-0000-0000-0000-000000000002")
        session_id = str(session_uuid)

        audio_payload = b"\x01\x02" * 320

        # Handshake frames returned by ws.recv() (2 times)
        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        session_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=150,
            session_id=session_id,
            payload=json.dumps({"dialog_id": "d"}).encode("utf-8"),
        )

        # Receiver iteration frames (after handshake)
        audio_frame = encode_frame(
            msg_type_bits=MSGTYPE_AUDIO_ONLY_SERVER,
            serialization_bits=SERIALIZATION_RAW,
            event=352,
            session_id=session_id,
            payload=audio_payload,
        )
        session_finished = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=152,
            session_id=session_id,
            payload=b"{}",
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, session_started])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        messages_iter = iter([audio_frame, session_finished])

        async def mock_anext(self):
            try:
                return next(messages_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = mock_anext

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        async def user_audio():
            yield b"\x00\x01" * 160

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()
            mod_websockets.ConnectionClosed = Exception

            with patch.object(mod.uuid, "uuid4", side_effect=[session_uuid, connect_uuid]):
                results = []
                async for event in plugin.converse_stream(user_audio()):
                    results.append(event)

        assert len(results) >= 1
        sent_events = [decode_frame(call.args[0]).event for call in mock_ws.send.await_args_list]
        assert DoubaoEvent.SAY_HELLO not in sent_events
        assert results[0].audio.data == audio_payload
        assert results[0].audio.is_final is False
        assert any(r.is_final for r in results)

    @pytest.mark.asyncio
    async def test_converse_stream_emits_barge_in_and_ids(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "zh_female_default",
                "system_prompt": "Hello",
            },
        )
        await plugin.initialize(config)

        session_uuid = uuid.UUID("00000000-0000-0000-0000-000000000021")
        connect_uuid = uuid.UUID("00000000-0000-0000-0000-000000000022")
        session_id = str(session_uuid)
        audio_payload = b"\x01\x02" * 320

        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        session_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=150,
            session_id=session_id,
            payload=json.dumps({"dialog_id": "d"}).encode("utf-8"),
        )
        asr_start = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=DoubaoEvent.ASR_START,
            session_id=session_id,
            payload=json.dumps({"question_id": "q1", "reply_id": "r1"}).encode("utf-8"),
        )
        llm_token = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=DoubaoEvent.LLM_TOKEN,
            session_id=session_id,
            payload=json.dumps({
                "content": "你好",
                "question_id": "q1",
                "reply_id": "r1",
            }).encode("utf-8"),
        )
        audio_frame = encode_frame(
            msg_type_bits=MSGTYPE_AUDIO_ONLY_SERVER,
            serialization_bits=SERIALIZATION_RAW,
            event=352,
            session_id=session_id,
            payload=audio_payload,
        )
        reply_done = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=DoubaoEvent.REPLY_DONE,
            session_id=session_id,
            payload=json.dumps({"question_id": "q1", "reply_id": "r1"}).encode("utf-8"),
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, session_started])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        messages_iter = iter([asr_start, llm_token, audio_frame, reply_done])

        async def mock_anext(self):
            try:
                return next(messages_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = mock_anext

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        async def user_audio():
            yield b"\x00\x01" * 160

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()
            mod_websockets.ConnectionClosed = Exception

            with patch.object(mod.uuid, "uuid4", side_effect=[session_uuid, connect_uuid]):
                results = []
                async for event in plugin.converse_stream(user_audio()):
                    results.append(event)

        assert results[0].barge_in is True
        assert results[0].question_id == "q1"
        assert results[0].reply_id == ""
        assert any(r.audio and r.reply_id == "r1" for r in results)
        assert any(r.is_final and r.question_id == "q1" and r.reply_id == "r1" for r in results)

    @pytest.mark.asyncio
    async def test_asr_start_finalizes_previous_reply_before_barge_in(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "zh_female_default",
                "system_prompt": "Hello",
            },
        )
        await plugin.initialize(config)

        session_uuid = uuid.UUID("00000000-0000-0000-0000-000000000031")
        connect_uuid = uuid.UUID("00000000-0000-0000-0000-000000000032")
        session_id = str(session_uuid)
        audio_payload = b"\x01\x02" * 320

        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        session_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=150,
            session_id=session_id,
            payload=json.dumps({"dialog_id": "d"}).encode("utf-8"),
        )
        llm_token = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=DoubaoEvent.LLM_TOKEN,
            session_id=session_id,
            payload=json.dumps({
                "content": "上一轮回答",
                "question_id": "q1",
                "reply_id": "r1",
            }).encode("utf-8"),
        )
        audio_frame = encode_frame(
            msg_type_bits=MSGTYPE_AUDIO_ONLY_SERVER,
            serialization_bits=SERIALIZATION_RAW,
            event=352,
            session_id=session_id,
            payload=audio_payload,
        )
        asr_start = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=DoubaoEvent.ASR_START,
            session_id=session_id,
            payload=json.dumps({"question_id": "q2"}).encode("utf-8"),
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, session_started])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        messages_iter = iter([llm_token, audio_frame, asr_start])

        async def mock_anext(self):
            try:
                return next(messages_iter)
            except StopIteration:
                raise StopAsyncIteration

        mock_ws.__aiter__ = lambda self: self
        mock_ws.__anext__ = mock_anext

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        async def user_audio():
            yield b"\x00\x01" * 160

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()
            mod_websockets.ConnectionClosed = Exception

            with patch.object(mod.uuid, "uuid4", side_effect=[session_uuid, connect_uuid]):
                results = []
                async for event in plugin.converse_stream(user_audio()):
                    results.append(event)

        final_index = next(
            i
            for i, event in enumerate(results)
            if event.is_final and event.question_id == "q1" and event.reply_id == "r1"
        )
        barge_index = next(i for i, event in enumerate(results) if event.barge_in)

        assert final_index < barge_index
        assert results[barge_index].question_id == "q2"
        assert results[barge_index].reply_id == ""

    @pytest.mark.asyncio
    async def test_check_voice_success(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "温柔文雅",
            },
        )
        await plugin.initialize(config)

        session_uuid = uuid.UUID("00000000-0000-0000-0000-000000000011")
        connect_uuid = uuid.UUID("00000000-0000-0000-0000-000000000012")
        session_id = str(session_uuid)

        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        session_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=150,
            session_id=session_id,
            payload=json.dumps({"dialog_id": "d"}).encode("utf-8"),
        )
        session_finished = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=152,
            session_id=session_id,
            payload=b"{}",
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, session_started, session_finished])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()

            with patch.object(mod.uuid, "uuid4", side_effect=[session_uuid, connect_uuid]):
                await plugin.check_voice(
                    VoiceLLMSessionConfig(voice="可爱女生")
                )

        sent_frames = [decode_frame(call.args[0]) for call in mock_ws.send.await_args_list]
        sent_events = [frame.event for frame in sent_frames]
        assert sent_events == [
            DoubaoEvent.START_CONNECTION,
            DoubaoEvent.START_SESSION,
            DoubaoEvent.FINISH_SESSION,
        ]
        start_payload = json.loads(
            decompress_payload(
                sent_frames[1].payload,
                sent_frames[1].compression_bits,
            )
        )
        assert start_payload["tts"]["speaker"] == SC20_VOICES["可爱女生"]

    @pytest.mark.asyncio
    async def test_shutdown(self):
        plugin = DoubaoRealtimePlugin()
        await plugin.shutdown()  # should not raise

    @pytest.mark.asyncio
    async def test_converse_stream_raises_on_start_session_error(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "S_knqs6GAT1",
                "system_prompt": "Hello",
            },
        )
        await plugin.initialize(config)

        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        start_error = self._error_frame(
            55000000,
            compress_payload(
                b'{"error":"resource ID is mismatched with speaker related resource"}',
                COMPRESSION_GZIP,
            ),
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, start_error])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        async def user_audio():
            if False:
                yield b""

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()
            mod_websockets.ConnectionClosed = type("FakeConnectionClosed", (Exception,), {})

            with pytest.raises(
                RuntimeError,
                match="resource ID is mismatched with speaker related resource",
            ):
                async for _ in plugin.converse_stream(user_audio()):
                    pass

    @pytest.mark.asyncio
    async def test_check_voice_raises_raw_provider_error_on_start_session_error(self):
        plugin = DoubaoRealtimePlugin()
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "key",
                "app_id": "app",
                "ws_url": "wss://test.example.com",
                "voice_type": "S_knqs6GAT1",
            },
        )
        await plugin.initialize(config)

        connection_started = encode_frame(
            msg_type_bits=MSGTYPE_FULL_SERVER,
            serialization_bits=SERIALIZATION_JSON,
            event=50,
            session_id=None,
            connect_id="connect-test",
            payload=b"{}",
        )
        start_error = self._error_frame(
            55000000,
            compress_payload(
                b'{"error":"resource ID is mismatched with speaker related resource"}',
                COMPRESSION_GZIP,
            ),
        )

        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()

        recv_iter = iter([connection_started, start_error])

        async def mock_recv():
            try:
                return next(recv_iter)
            except StopIteration:
                raise RuntimeError("unexpected extra ws.recv() call")

        mock_ws.recv = AsyncMock(side_effect=mock_recv)

        class MockWSContext:
            async def __aenter__(self):
                return mock_ws

            async def __aexit__(self, *args):
                return None

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import inference.plugins.voice_llm.doubao_realtime as mod

            mod_websockets = __import__("sys").modules["websockets"]
            mod_websockets.connect.return_value = MockWSContext()

            with pytest.raises(
                RuntimeError,
                match=r'^\{"error":"resource ID is mismatched with speaker related resource"\}$',
            ):
                await plugin.check_voice()


class TestDoubaoSessionConfigOverrides:
    """Tests for per-session config merging via with_overrides()."""

    def _base_config(self) -> DoubaoSessionConfig:
        return DoubaoSessionConfig(
            access_token="tok",
            app_id="app",
            voice_type="温柔文雅",
            bot_name="豆包",
            system_prompt="默认提示词",
            speaking_style="默认风格",
            say_hello_content="默认欢迎语",
        )

    def test_empty_overrides_returns_same(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig()
        result = base.with_overrides(session)
        assert result is base  # same object, no copy

    def test_voice_override(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(voice="可爱女生")
        result = base.with_overrides(session)
        assert result.voice_type == "可爱女生"
        assert result.bot_name == "豆包"  # unchanged

    def test_all_overrides(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(
            voice="性感御姐",
            bot_name="小助手",
            system_prompt="你是一个温柔的助手",
            speaking_style="慢速说话",
            welcome_message="你好呀！",
        )
        result = base.with_overrides(session)
        assert result.voice_type == "性感御姐"
        assert result.bot_name == "小助手"
        assert result.system_prompt == "你是一个温柔的助手"
        assert result.speaking_style == "慢速说话"
        assert result.say_hello_content == "你好呀！"
        # credentials unchanged
        assert result.access_token == "tok"
        assert result.app_id == "app"

    def test_base_config_unchanged_after_override(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(voice="可爱女生", bot_name="新名字")
        _ = base.with_overrides(session)
        assert base.voice_type == "温柔文雅"
        assert base.bot_name == "豆包"

    def test_speaker_resolution_with_override(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(voice="可爱女生")
        result = base.with_overrides(session)
        payload = result.build_start_session_payload()
        assert payload["tts"]["speaker"] == SC20_VOICES["可爱女生"]

    def test_custom_speaker_id_passthrough(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(voice="S_123456")
        result = base.with_overrides(session)
        payload = result.build_start_session_payload()
        assert payload["tts"]["speaker"] == "S_123456"

    def test_welcome_message_override(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(welcome_message="嗨！我是你的助手")
        result = base.with_overrides(session)
        payload = result.build_say_hello_payload()
        assert payload["content"] == "嗨！我是你的助手"

    def test_empty_welcome_message_clears_base_default(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(welcome_message="")
        result = base.with_overrides(session)
        assert result is not base
        assert result.say_hello_content == ""

    def test_dialog_context_override_in_start_session_payload(self):
        base = self._base_config()
        session = VoiceLLMSessionConfig(
            session_id="local-conversation",
            dialog_context=[
                VoiceLLMDialogContextItem(
                    role="user",
                    text="我叫小明",
                    timestamp=1777608000000,
                ),
                VoiceLLMDialogContextItem(
                    role="assistant",
                    text="我记住了，你叫小明。",
                    timestamp=1777608001000,
                ),
            ],
        )

        result = base.with_overrides(session)
        payload = result.build_start_session_payload(dialog_id="provider-dialog")

        assert payload["dialog"]["dialog_id"] == "provider-dialog"
        assert payload["dialog"]["dialog_context"] == [
            {
                "role": "user",
                "text": "我叫小明",
                "timestamp": 1777608000000,
            },
            {
                "role": "assistant",
                "text": "我记住了，你叫小明。",
                "timestamp": 1777608001000,
            },
        ]

    def test_plugin_config_defaults_to_no_welcome_message(self):
        config = PluginConfig(
            plugin_name="voice_llm.doubao",
            params={
                "access_token": "tok",
                "app_id": "app",
            },
        )
        result = DoubaoSessionConfig.from_plugin_config(config)
        assert result.say_hello_content == ""
        assert result.has_welcome_message is False
