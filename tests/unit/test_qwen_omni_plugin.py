import asyncio
import base64
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from inference.core.types import (
    ImageFrame,
    PluginConfig,
    ToolResult,
    VoiceLLMInputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.persona_agent import PERSONA_TOOL_DEFINITIONS
from inference.plugins.voice_llm.qwen_omni_realtime import (
    QwenOmniRealtimePlugin,
    _QwenResponseCoordinator,
)


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


async def wait_for_sent(ws: FakeQwenWS, count: int) -> None:
    for _ in range(50):
        if len(ws.sent) >= count:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"expected at least {count} sent events, got {len(ws.sent)}")


@pytest.mark.asyncio
async def test_initialize_sets_qwen_omni_defaults():
    plugin = QwenOmniRealtimePlugin()

    await plugin.initialize(
        PluginConfig(
            plugin_name="omni.qwen_omni",
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
    assert plugin.system_prompt == ""
    assert plugin.ws_url.endswith("model=qwen3.5-omni-flash-realtime")


@pytest.mark.asyncio
async def test_check_voice_configures_session_with_voice_override():
    plugin = QwenOmniRealtimePlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="omni.qwen_omni",
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
    assert session["instructions"] == ""
    assert session["turn_detection"]["type"] == "semantic_vad"


@pytest.mark.asyncio
async def test_initialize_keeps_qwen_omni_system_prompt_when_configured():
    plugin = QwenOmniRealtimePlugin()

    await plugin.initialize(
        PluginConfig(
            plugin_name="omni.qwen_omni",
            params={
                "api_key": "dashscope-key",
                "system_prompt": "角色背景",
            },
        )
    )

    assert plugin.system_prompt == "角色背景"


@pytest.mark.asyncio
async def test_converse_stream_emits_audio_transcripts_and_final():
    plugin = QwenOmniRealtimePlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="omni.qwen_omni",
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


@pytest.mark.asyncio
async def test_converse_stream_text_waits_for_response_before_closing_ws():
    plugin = QwenOmniRealtimePlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="omni.qwen_omni",
            params={"api_key": "dashscope-key"},
        )
    )
    audio_bytes = b"\x01\x00\x02\x00"
    ws = FakeQwenWS(
        [
            {"type": "session.updated"},
            {"type": "response.created", "response": {"id": "resp-1"}},
            {"type": "response.audio_transcript.delta", "delta": "欢迎回来"},
            {
                "type": "response.audio.delta",
                "delta": base64.b64encode(audio_bytes).decode("ascii"),
            },
            {"type": "response.audio_transcript.done", "transcript": "欢迎回来"},
            {"type": "response.done"},
        ]
    )
    websockets = SimpleNamespace(connect=AsyncMock(return_value=ws))

    async def inputs():
        yield VoiceLLMInputEvent(text="请主动打招呼。")

    with patch.dict("sys.modules", {"websockets": websockets}):
        outputs = [
            event
            async for event in plugin.converse_stream(
                inputs(),
                VoiceLLMSessionConfig(session_id="session-1", voice="Tina"),
            )
        ]

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["session.update", "conversation.item.create", "response.create"]
    assert ws.closed is True
    assert outputs[0].transcript == "欢迎回来"
    assert outputs[1].audio is not None
    assert outputs[1].audio.data == audio_bytes
    assert outputs[2].is_final is True
    assert outputs[2].transcript == "欢迎回来"


@pytest.mark.asyncio
async def test_send_inputs_sends_image_after_audio():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])
    image_bytes = b"\xff\xd8\xff\x00"

    async def inputs():
        yield VoiceLLMInputEvent(audio=b"\x03\x00")
        yield VoiceLLMInputEvent(
            image=ImageFrame(
                data=image_bytes,
                mime_type="image/jpeg",
                width=640,
                height=360,
                source="camera",
                frame_seq=1,
            )
        )

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["input_audio_buffer.append", "input_image_buffer.append"]
    assert base64.b64decode(ws.sent[0]["audio"]) == b"\x03\x00"
    assert base64.b64decode(ws.sent[1]["image"]) == image_bytes


@pytest.mark.asyncio
async def test_send_inputs_buffers_latest_image_until_first_audio():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])
    old_image = b"\xff\xd8\xffold"
    latest_image = b"\xff\xd8\xfflatest"

    async def inputs():
        yield VoiceLLMInputEvent(
            image=ImageFrame(data=old_image, mime_type="image/jpeg", frame_seq=1)
        )
        yield VoiceLLMInputEvent(
            image=ImageFrame(data=latest_image, mime_type="image/jpeg", frame_seq=2)
        )
        yield VoiceLLMInputEvent(audio=b"\x03\x00")

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["input_audio_buffer.append", "input_image_buffer.append"]
    assert base64.b64decode(ws.sent[1]["image"]) == latest_image


@pytest.mark.asyncio
async def test_send_inputs_sends_text_message_and_response_create():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])

    async def inputs():
        yield VoiceLLMInputEvent(text="后台任务结果已经返回。请回复用户。")

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["conversation.item.create", "response.create"]
    assert ws.sent[0]["item"]["type"] == "message"
    assert ws.sent[0]["item"]["role"] == "user"
    assert ws.sent[0]["item"]["content"] == [
        {"type": "input_text", "text": "后台任务结果已经返回。请回复用户。"}
    ]
    assert ws.sent[1]["response"] == {"modalities": ["text", "audio"]}


@pytest.mark.asyncio
async def test_send_inputs_sends_response_instructions_create():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])

    async def inputs():
        yield VoiceLLMInputEvent(response_instructions="依据角色素材回答。")

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    assert [event["type"] for event in ws.sent] == ["response.create"]
    assert ws.sent[0]["response"] == {
        "modalities": ["text", "audio"],
        "instructions": "依据角色素材回答。",
    }


@pytest.mark.asyncio
async def test_deferred_response_speech_start_keeps_response_coordinator_idle():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([{"type": "input_audio_buffer.speech_started"}])
    output_queue = asyncio.Queue()
    coordinator = _QwenResponseCoordinator()

    await plugin._receive_events(
        ws,
        "session-1",
        output_queue,
        coordinator,
        defer_response=True,
    )

    first = await output_queue.get()
    assert first.barge_in is True
    assert await asyncio.wait_for(coordinator.wait_idle(), timeout=0.1)


@pytest.mark.asyncio
async def test_send_inputs_defers_text_response_until_active_response_done():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])
    response_coordinator = _QwenResponseCoordinator()
    await response_coordinator.mark_response_started()
    inputs_queue = asyncio.Queue()

    async def inputs():
        while True:
            event = await inputs_queue.get()
            if event is None:
                return
            yield event

    sender = asyncio.create_task(
        plugin._send_inputs(
            ws,
            inputs(),
            "session-1",
            asyncio.Queue(),
            response_coordinator,
        )
    )

    await inputs_queue.put(VoiceLLMInputEvent(text="后台任务结果已经返回。请回复用户。"))
    await asyncio.sleep(0.02)
    assert ws.sent == []

    await inputs_queue.put(VoiceLLMInputEvent(audio=b"\x03\x00"))
    await wait_for_sent(ws, 1)
    assert [event["type"] for event in ws.sent] == ["input_audio_buffer.append"]

    await response_coordinator.mark_response_done()
    await wait_for_sent(ws, 3)
    assert [event["type"] for event in ws.sent] == [
        "input_audio_buffer.append",
        "conversation.item.create",
        "response.create",
    ]
    assert ws.sent[1]["item"]["role"] == "user"
    assert ws.sent[2]["response"] == {"modalities": ["text", "audio"]}

    await inputs_queue.put(None)
    await sender


def test_session_payload_includes_hidden_tools():
    plugin = QwenOmniRealtimePlugin()
    plugin.enable_search = True
    plugin.search_options = {"enable_source": True}

    payload = plugin._session_payload(
        VoiceLLMSessionConfig(
            tools=PERSONA_TOOL_DEFINITIONS,
        )
    )

    assert "tool_choice" not in payload
    assert "enable_search" not in payload
    assert "search_options" not in payload
    create_task_tool = next(tool for tool in payload["tools"] if tool["function"]["name"] == "create_task")
    assert create_task_tool["type"] == "function"
    assert create_task_tool["function"]["parameters"]["required"] == ["description"]
    assert set(create_task_tool["function"]["parameters"]["properties"]) == {"description"}
    assert create_task_tool["function"]["parameters"]["properties"]["description"]["type"] == "string"


def test_session_payload_can_defer_automatic_response():
    plugin = QwenOmniRealtimePlugin()

    payload = plugin._session_payload(VoiceLLMSessionConfig(defer_response=True))

    assert payload["turn_detection"]["create_response"] is False


def test_session_payload_keeps_search_when_tools_absent():
    plugin = QwenOmniRealtimePlugin()
    plugin.enable_search = True
    plugin.search_options = {"enable_source": True}

    payload = plugin._session_payload(VoiceLLMSessionConfig())

    assert payload["enable_search"] is True
    assert payload["search_options"] == {"enable_source": True}


def test_model_event_logs_keep_stream_deltas_out_of_info(caplog):
    logger_name = "inference.plugins.voice_llm.qwen_omni_realtime"

    with caplog.at_level(logging.INFO, logger=logger_name):
        QwenOmniRealtimePlugin._log_server_event(
            "session-1",
            {"type": "response.audio_transcript.delta", "delta": "收到"},
        )
        QwenOmniRealtimePlugin._log_server_event(
            "session-1",
            {"type": "response.audio_transcript.done", "transcript": "收到"},
        )
        QwenOmniRealtimePlugin._log_server_event(
            "session-1",
            {"type": "response.audio.delta", "delta": "abcd"},
        )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "response.audio_transcript.done" in messages[0]
    assert "收到" in messages[0]


@pytest.mark.asyncio
async def test_send_inputs_sends_tool_result_and_response_create():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])

    async def inputs():
        yield VoiceLLMInputEvent(
            tool_result=ToolResult(
                id="call-1",
                name="create_task",
                result={"id": "task-1", "status": "queued"},
            )
        )

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["conversation.item.create", "response.create"]
    assert ws.sent[0]["item"]["type"] == "function_call_output"
    assert ws.sent[0]["item"]["call_id"] == "call-1"
    assert json.loads(ws.sent[0]["item"]["output"]) == {"id": "task-1", "status": "queued"}


@pytest.mark.asyncio
async def test_send_inputs_sends_tool_output_immediately_but_defers_response_create():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])
    response_coordinator = _QwenResponseCoordinator()
    await response_coordinator.mark_response_started()
    inputs_queue = asyncio.Queue()

    async def inputs():
        while True:
            event = await inputs_queue.get()
            if event is None:
                return
            yield event

    sender = asyncio.create_task(
        plugin._send_inputs(
            ws,
            inputs(),
            "session-1",
            asyncio.Queue(),
            response_coordinator,
        )
    )

    await inputs_queue.put(
        VoiceLLMInputEvent(
            tool_result=ToolResult(
                id="call-1",
                name="create_task",
                result={"id": "task-1", "status": "queued"},
            )
        )
    )
    await wait_for_sent(ws, 1)
    assert [event["type"] for event in ws.sent] == ["conversation.item.create"]
    assert ws.sent[0]["item"]["type"] == "function_call_output"

    await response_coordinator.mark_response_done()
    await wait_for_sent(ws, 2)
    assert [event["type"] for event in ws.sent] == [
        "conversation.item.create",
        "response.create",
    ]

    await inputs_queue.put(None)
    await sender


@pytest.mark.asyncio
async def test_active_response_error_retries_response_create_without_duplicate_item():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])
    response_coordinator = _QwenResponseCoordinator()
    inputs_queue = asyncio.Queue()

    async def inputs():
        while True:
            event = await inputs_queue.get()
            if event is None:
                return
            yield event

    sender = asyncio.create_task(
        plugin._send_inputs(
            ws,
            inputs(),
            "session-1",
            asyncio.Queue(),
            response_coordinator,
        )
    )

    await inputs_queue.put(VoiceLLMInputEvent(text="后台任务结果已经返回。请回复用户。"))
    await wait_for_sent(ws, 2)
    await response_coordinator.mark_active_response_error()
    await asyncio.sleep(0.02)
    assert [event["type"] for event in ws.sent] == [
        "conversation.item.create",
        "response.create",
    ]

    await response_coordinator.mark_response_done()
    await wait_for_sent(ws, 3)
    assert [event["type"] for event in ws.sent] == [
        "conversation.item.create",
        "response.create",
        "response.create",
    ]

    await inputs_queue.put(None)
    await sender


@pytest.mark.asyncio
async def test_send_inputs_suppresses_response_create_for_wait_tool_result():
    plugin = QwenOmniRealtimePlugin()
    ws = FakeQwenWS([])

    async def inputs():
        yield VoiceLLMInputEvent(
            tool_result=ToolResult(
                id="wait-1",
                name="wait_for_more_input",
                result={"ok": True, "waiting": True},
                suppress_response=True,
            )
        )

    await plugin._send_inputs(ws, inputs(), "session-1", asyncio.Queue())

    sent_types = [event["type"] for event in ws.sent]
    assert sent_types == ["conversation.item.create"]
    assert ws.sent[0]["item"]["type"] == "function_call_output"
    assert ws.sent[0]["item"]["call_id"] == "wait-1"
    assert json.loads(ws.sent[0]["item"]["output"]) == {"ok": True, "waiting": True}


@pytest.mark.asyncio
async def test_receive_events_emits_tool_call():
    plugin = QwenOmniRealtimePlugin()
    output_queue = asyncio.Queue()
    ws = FakeQwenWS(
        [
            {
                "type": "response.function_call_arguments.done",
                "call_id": "call-1",
                "name": "create_task",
                "arguments": json.dumps({"user_request": "今天知乎有哪些热门信息"}, ensure_ascii=False),
            }
        ]
    )

    await plugin._receive_events(ws, "session-1", output_queue)

    event = await output_queue.get()
    assert event.tool_calls[0].id == "call-1"
    assert event.tool_calls[0].name == "create_task"
    assert event.tool_calls[0].arguments == {"user_request": "今天知乎有哪些热门信息"}
    assert await output_queue.get() is None
