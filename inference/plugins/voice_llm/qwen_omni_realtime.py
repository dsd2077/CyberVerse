import asyncio
import base64
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator

from inference.core.types import (
    AudioChunk,
    ImageFrame,
    PluginConfig,
    ToolCall,
    ToolResult,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.qwen_endpoint import dashscope_realtime_ws_url
from inference.plugins.voice_llm.base import VoiceCheckError, VoiceLLMPlugin

logger = logging.getLogger(__name__)
_MAX_IMAGE_BYTES = 500 * 1024


class QwenOmniRealtimePlugin(VoiceLLMPlugin):
    """DashScope Qwen Omni realtime omni model plugin."""

    name = "omni.qwen_omni"

    def __init__(self) -> None:
        self.api_key = ""
        self.model = "qwen3.5-omni-flash-realtime"
        self.ws_url = ""
        self.voice = "Tina"
        self.system_prompt = ""
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.vad_type = "semantic_vad"
        self.vad_threshold = 0.5
        self.vad_silence_duration_ms = 800
        self.enable_search: bool | None = None
        self.search_options: dict[str, Any] | None = None
        self.temperature: float | None = None
        self.top_p: float | None = None
        self.top_k: int | None = None
        self.max_tokens: int | None = None
        self._active_ws: Any | None = None

    async def initialize(self, config: PluginConfig) -> None:
        params = config.params
        self.api_key = params.get("api_key", self.api_key)
        self.model = params.get("model", self.model)
        self.ws_url = dashscope_realtime_ws_url(self.model, "DASHSCOPE_OMNI_WS_URL")
        self.voice = params.get("voice", self.voice)
        self.system_prompt = params.get("system_prompt", self.system_prompt)
        self.input_sample_rate = int(
            params.get("input_sample_rate", self.input_sample_rate)
        )
        self.output_sample_rate = int(
            params.get("output_sample_rate", self.output_sample_rate)
        )
        self.vad_type = params.get("vad_type", self.vad_type)
        self.vad_threshold = float(params.get("vad_threshold", self.vad_threshold))
        self.vad_silence_duration_ms = int(
            params.get("vad_silence_duration_ms", self.vad_silence_duration_ms)
        )
        self.enable_search = self._optional_bool(params.get("enable_search"))
        search_options = params.get("search_options")
        if isinstance(search_options, dict):
            self.search_options = search_options
        self.temperature = self._optional_float(params.get("temperature"))
        self.top_p = self._optional_float(params.get("top_p"))
        self.top_k = self._optional_int(params.get("top_k"))
        self.max_tokens = self._optional_int(params.get("max_tokens"))

    async def check_voice(
        self,
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> None:
        import websockets

        ws = await self._connect(websockets)
        try:
            await self._configure_session(ws, session_config or VoiceLLMSessionConfig())
        except RuntimeError as exc:
            raise VoiceCheckError(str(exc)) from exc
        finally:
            await ws.close()

    async def converse_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        import websockets

        config = session_config or VoiceLLMSessionConfig()
        ws = await self._connect(websockets)
        self._active_ws = ws
        response_coordinator = _QwenResponseCoordinator()
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None] = (
            asyncio.Queue()
        )
        sender_task: asyncio.Task | None = None
        receiver_task: asyncio.Task | None = None
        try:
            await self._configure_session(ws, config)
            sender_task = asyncio.create_task(
                self._send_inputs(
                    ws,
                    input_stream,
                    config.session_id,
                    output_queue,
                    response_coordinator,
                    drain_responses_on_close=True,
                )
            )
            receiver_task = asyncio.create_task(
                self._receive_events(
                    ws,
                    config.session_id,
                    output_queue,
                    response_coordinator,
                )
            )

            while True:
                item = await output_queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            for task in (sender_task, receiver_task):
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            if self._active_ws is ws:
                self._active_ws = None
            await ws.close()

    async def interrupt(self) -> None:
        ws = self._active_ws
        if ws is None:
            return
        for event_type in ("response.cancel", "input_audio_buffer.clear"):
            try:
                await self._send_json(
                    ws,
                    {
                        "type": event_type,
                        "event_id": self._event_id("qwen_omni", "interrupt"),
                    },
                )
            except Exception:
                logger.debug("Failed to send Qwen Omni interrupt event", exc_info=True)

    async def _connect(self, websockets: Any):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            return await websockets.connect(
                self.ws_url,
                additional_headers=headers,
            )
        except TypeError:
            return await websockets.connect(
                self.ws_url,
                extra_headers=headers,
            )

    async def _configure_session(
        self,
        ws: Any,
        session_config: VoiceLLMSessionConfig,
    ) -> None:
        await self._send_json(
            ws,
            {
                "type": "session.update",
                "event_id": self._event_id(session_config.session_id, "session"),
                "session": self._session_payload(session_config),
            },
        )

        while True:
            event = self._decode_message(await ws.recv())
            event_type = event.get("type", "")
            if event_type in {"session.created", "session.updated"}:
                return
            if event_type == "error":
                raise RuntimeError(self._error_message(event))

    async def _send_inputs(
        self,
        ws: Any,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_id: str,
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None],
        response_coordinator: "_QwenResponseCoordinator | None" = None,
        drain_responses_on_close: bool = False,
    ) -> None:
        response_coordinator = response_coordinator or _QwenResponseCoordinator()
        response_sender_task = asyncio.create_task(
            self._send_deferred_responses(ws, response_coordinator, output_queue)
        )
        try:
            pending_image: ImageFrame | None = None
            has_sent_audio = False
            expects_deferred_response = False
            async for event in input_stream:
                if event.tool_result:
                    await self._send_tool_result(
                        ws,
                        session_id,
                        event.tool_result,
                        response_coordinator,
                    )
                    if not event.tool_result.suppress_response:
                        expects_deferred_response = True
                    continue
                if event.text:
                    expects_deferred_response = True
                    await self._send_text(ws, session_id, event.text, response_coordinator)
                    continue
                if event.audio:
                    has_sent_audio = True
                    await self._send_json(
                        ws,
                        {
                            "type": "input_audio_buffer.append",
                            "event_id": self._event_id(session_id, "audio"),
                            "audio": base64.b64encode(event.audio).decode("ascii"),
                        },
                    )
                    # Keep image strictly after an audio append. This avoids provider-side
                    # ordering violations when a new turn starts and an image arrives first.
                    if pending_image is not None:
                        await self._send_image(ws, session_id, pending_image)
                        pending_image = None
                if event.image is not None:
                    if not self._valid_image(event.image):
                        continue
                    # Always buffer the latest valid frame and flush it only after
                    # the next audio chunk is appended.
                    pending_image = event.image
            if pending_image is not None and has_sent_audio:
                # If stream ends after image input, flush once to avoid dropping
                # the latest frame while still guaranteeing audio-first ordering.
                await self._send_image(ws, session_id, pending_image)
            if drain_responses_on_close and expects_deferred_response:
                await response_coordinator.wait_all_responses_done(timeout=60.0)
        except Exception as exc:
            await output_queue.put(exc)
        finally:
            await response_coordinator.close()
            if not response_sender_task.done():
                try:
                    await response_sender_task
                except asyncio.CancelledError:
                    pass
            try:
                await ws.close()
            except Exception:
                pass

    async def _send_image(self, ws: Any, session_id: str, image: ImageFrame) -> None:
        await self._send_json(
            ws,
            {
                "type": "input_image_buffer.append",
                "event_id": self._event_id(session_id, "image"),
                "image": base64.b64encode(image.data).decode("ascii"),
            },
        )

    async def _send_text(
        self,
        ws: Any,
        session_id: str,
        text: str,
        response_coordinator: "_QwenResponseCoordinator",
    ) -> None:
        await response_coordinator.enqueue(
            _QwenDeferredResponse(
                item_payload={
                    "type": "conversation.item.create",
                    "event_id": self._event_id(session_id, "text"),
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text,
                            }
                        ],
                    },
                },
                response_payload={
                    "type": "response.create",
                    "event_id": self._event_id(session_id, "text_response"),
                    "response": {"modalities": ["text", "audio"]},
                },
            )
        )

    async def _send_tool_result(
        self,
        ws: Any,
        session_id: str,
        result: ToolResult,
        response_coordinator: "_QwenResponseCoordinator",
    ) -> None:
        await self._send_json(
            ws,
            {
                "type": "conversation.item.create",
                "event_id": self._event_id(session_id, "tool_result"),
                "item": {
                    "type": "function_call_output",
                    "call_id": result.id,
                    "output": json.dumps(result.result, ensure_ascii=False),
                },
            },
        )
        if result.suppress_response:
            return
        await response_coordinator.enqueue(
            _QwenDeferredResponse(
                response_payload={
                    "type": "response.create",
                    "event_id": self._event_id(session_id, "tool_response"),
                }
            )
        )

    async def _send_deferred_responses(
        self,
        ws: Any,
        response_coordinator: "_QwenResponseCoordinator",
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None],
    ) -> None:
        try:
            while True:
                request = await response_coordinator.next_request()
                if request is None:
                    return
                await self._send_deferred_response(ws, response_coordinator, request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await output_queue.put(exc)

    async def _send_deferred_response(
        self,
        ws: Any,
        response_coordinator: "_QwenResponseCoordinator",
        request: "_QwenDeferredResponse",
    ) -> None:
        if not await response_coordinator.wait_idle():
            return
        if request.item_payload is not None and not request.item_sent:
            await self._send_json(ws, request.item_payload)
            request.item_sent = True
        await response_coordinator.begin_client_response(request)
        try:
            await self._send_json(ws, request.response_payload)
        except Exception:
            await response_coordinator.release_client_response(request)
            raise

    @staticmethod
    def _valid_image(image: ImageFrame) -> bool:
        mime_type = (image.mime_type or "").lower()
        if mime_type and mime_type not in {"image/jpeg", "image/jpg"}:
            return False
        data = image.data or b""
        if len(data) == 0 or len(data) > _MAX_IMAGE_BYTES:
            return False
        return len(data) >= 3 and data[0] == 0xFF and data[1] == 0xD8 and data[2] == 0xFF

    async def _receive_events(
        self,
        ws: Any,
        session_id: str,
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None],
        response_coordinator: "_QwenResponseCoordinator | None" = None,
    ) -> None:
        response_coordinator = response_coordinator or _QwenResponseCoordinator()
        turn_state = _QwenTurnState(session_id=session_id or "qwen_omni")
        tool_arg_parts: dict[str, str] = {}
        emitted_tool_calls: set[str] = set()
        try:
            async for message in ws:
                event = self._decode_message(message)
                self._log_server_event(session_id, event)
                event_type = event.get("type", "")
                if event_type == "error":
                    message = self._error_message(event)
                    if self._is_active_response_error(message):
                        logger.info(
                            "qwen_omni deferred response delayed by active response session=%s",
                            session_id or "qwen_omni",
                        )
                        await response_coordinator.mark_active_response_error()
                        continue
                    raise RuntimeError(message)
                if event_type in {"session.created", "session.updated"}:
                    continue

                if event_type == "response.function_call_arguments.delta":
                    call_id = str(event.get("call_id") or event.get("item_id") or "")
                    if call_id:
                        tool_arg_parts[call_id] = tool_arg_parts.get(call_id, "") + str(event.get("delta", "") or "")
                    continue

                if event_type == "response.function_call_arguments.done":
                    call = self._tool_call_from_event(event, tool_arg_parts)
                    if call and call.id not in emitted_tool_calls:
                        emitted_tool_calls.add(call.id)
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                tool_calls=[call],
                                question_id=turn_state.question_id,
                                reply_id=turn_state.reply_id,
                            )
                        )
                    continue

                if event_type == "response.output_item.done":
                    item = event.get("item")
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        call = self._tool_call_from_event(item, tool_arg_parts)
                        if call and call.id not in emitted_tool_calls:
                            emitted_tool_calls.add(call.id)
                            await output_queue.put(
                                VoiceLLMOutputEvent(
                                    tool_calls=[call],
                                    question_id=turn_state.question_id,
                                    reply_id=turn_state.reply_id,
                                )
                            )
                    continue

                if event_type == "input_audio_buffer.speech_started":
                    await response_coordinator.mark_response_started()
                    turn_state.start_next_turn()
                    await output_queue.put(
                        VoiceLLMOutputEvent(
                            barge_in=True,
                            question_id=turn_state.question_id,
                            reply_id=turn_state.reply_id,
                        )
                    )
                    continue

                if event_type == "response.created":
                    await response_coordinator.mark_response_started()
                    response = event.get("response")
                    if isinstance(response, dict):
                        response_id = str(response.get("id", "") or "")
                        if response_id and not turn_state.question_id:
                            turn_state.start_next_turn()
                            turn_state.reply_id = response_id
                    continue

                if event_type == "conversation.item.input_audio_transcription.completed":
                    turn_state.ensure_turn()
                    transcript = str(event.get("transcript", "") or "").strip()
                    if transcript:
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                user_transcript=transcript,
                                question_id=turn_state.question_id,
                                reply_id=turn_state.reply_id,
                            )
                        )
                    continue

                if event_type == "response.audio_transcript.delta":
                    turn_state.ensure_turn()
                    delta = str(event.get("delta", "") or "")
                    if delta:
                        turn_state.assistant_text += delta
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                transcript=delta,
                                question_id=turn_state.question_id,
                                reply_id=turn_state.reply_id,
                            )
                        )
                    continue

                if event_type == "response.audio_transcript.done":
                    transcript = str(event.get("transcript", "") or "")
                    if transcript:
                        turn_state.assistant_text = transcript
                    continue

                if event_type == "response.audio.delta":
                    turn_state.ensure_turn()
                    delta = str(event.get("delta", "") or "")
                    if not delta:
                        continue
                    audio_payload = base64.b64decode(delta)
                    if audio_payload:
                        turn_state.has_audio = True
                    await output_queue.put(
                        VoiceLLMOutputEvent(
                            audio=AudioChunk(
                                data=audio_payload,
                                sample_rate=self.output_sample_rate,
                                channels=1,
                                format="pcm_s16le",
                            ),
                            question_id=turn_state.question_id,
                            reply_id=turn_state.reply_id,
                        )
                    )
                    continue

                if event_type == "response.done":
                    if turn_state.has_content:
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                audio=AudioChunk(
                                    data=b"",
                                    sample_rate=self.output_sample_rate,
                                    channels=1,
                                    format="pcm_s16le",
                                    is_final=True,
                                )
                                if turn_state.has_audio
                                else None,
                                transcript=turn_state.assistant_text,
                                is_final=True,
                                question_id=turn_state.question_id,
                                reply_id=turn_state.reply_id,
                            )
                        )
                    turn_state.reset()
                    await response_coordinator.mark_response_done()
                    continue
        except Exception as exc:
            if not getattr(ws, "closed", False):
                await output_queue.put(exc)
        finally:
            await output_queue.put(None)

    def _session_payload(self, session_config: VoiceLLMSessionConfig) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "modalities": ["text", "audio"],
            "voice": session_config.voice or self.voice,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            "instructions": self._instructions(session_config),
            "turn_detection": {
                "type": self.vad_type,
                "threshold": self.vad_threshold,
                "silence_duration_ms": self.vad_silence_duration_ms,
            },
        }
        has_tools = bool(session_config.tools)
        optional_values: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        if not has_tools:
            optional_values["enable_search"] = self.enable_search
            optional_values["search_options"] = self.search_options
        for key, value in optional_values.items():
            if value is not None:
                payload[key] = value
        if has_tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                }
                for tool in session_config.tools
            ]
        return payload

    def _instructions(self, session_config: VoiceLLMSessionConfig) -> str:
        parts: list[str] = []
        if session_config.bot_name:
            parts.append(f"名字：{session_config.bot_name}")
        parts.append(session_config.system_prompt or self.system_prompt)
        if session_config.speaking_style:
            parts.append(f"说话风格：{session_config.speaking_style}")
        if session_config.dialog_context:
            parts.append("以下是最近的对话上下文，请在回答时保持连续性：")
            for item in session_config.dialog_context:
                role = "用户" if item.role == "user" else "助手"
                parts.append(f"{role}：{item.text}")
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _tool_call_from_event(event: dict[str, Any], arg_parts: dict[str, str]) -> ToolCall | None:
        call_id = str(event.get("call_id") or event.get("id") or event.get("item_id") or "")
        name = str(event.get("name") or "")
        raw_args = event.get("arguments")
        if raw_args is None and call_id:
            raw_args = arg_parts.get(call_id, "")
        if not call_id or not name:
            return None
        return ToolCall(
            id=call_id,
            name=name,
            arguments=QwenOmniRealtimePlugin._parse_tool_arguments(raw_args),
        )

    @staticmethod
    def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if raw is None:
            return {}
        try:
            parsed = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _clip_text(value: Any, limit: int = 180) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @classmethod
    def _server_event_log_fields(cls, event: dict[str, Any]) -> dict[str, Any]:
        event_type = str(event.get("type") or "")
        fields: dict[str, Any] = {}
        for key in ("response_id", "item_id", "call_id", "name", "output_index"):
            if key in event and event.get(key) not in (None, ""):
                fields[key] = event.get(key)
        if event_type == "response.audio.delta":
            fields["audio_delta_b64_len"] = len(str(event.get("delta") or ""))
        elif event_type in {"response.audio_transcript.delta", "response.function_call_arguments.delta"}:
            fields["delta"] = cls._clip_text(event.get("delta"))
        if "transcript" in event:
            fields["transcript"] = cls._clip_text(event.get("transcript"))
        if "arguments" in event:
            fields["arguments"] = cls._clip_text(event.get("arguments"))
        item = event.get("item")
        if isinstance(item, dict):
            item_fields = {
                key: item.get(key)
                for key in ("type", "id", "call_id", "name")
                if item.get(key) not in (None, "")
            }
            if "arguments" in item:
                item_fields["arguments"] = cls._clip_text(item.get("arguments"))
            fields["item"] = item_fields
        response = event.get("response")
        if isinstance(response, dict):
            fields["response"] = {
                key: response.get(key)
                for key in ("id", "status")
                if response.get(key) not in (None, "")
            }
        error = event.get("error")
        if error:
            fields["error"] = cls._clip_text(error)
        return fields

    @classmethod
    def _server_event_log_level(cls, event: dict[str, Any]) -> int:
        event_type = str(event.get("type") or "")
        if event_type == "error":
            return logging.ERROR
        if event_type in {
            "session.created",
            "session.updated",
            "input_audio_buffer.speech_started",
            "conversation.item.input_audio_transcription.completed",
            "response.created",
            "response.audio_transcript.done",
            "response.function_call_arguments.done",
            "response.done",
        }:
            return logging.INFO
        return logging.DEBUG

    @classmethod
    def _log_server_event(cls, session_id: str, event: dict[str, Any]) -> None:
        event_type = str(event.get("type") or "unknown")
        level = cls._server_event_log_level(event)
        if not logger.isEnabledFor(level):
            return
        fields = cls._server_event_log_fields(event)
        logger.log(
            level,
            "qwen_omni model event session=%s type=%s fields=%s",
            session_id or "qwen_omni",
            event_type,
            json.dumps(fields, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    async def _send_json(ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _decode_message(message: str | bytes) -> dict[str, Any]:
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        return json.loads(message)

    @staticmethod
    def _event_id(session_id: str, suffix: str) -> str:
        base = session_id or "qwen_omni"
        return f"{base}_{suffix}_{int(time.time() * 1000)}"

    @staticmethod
    def _error_message(event: dict[str, Any]) -> str:
        error = event.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or error.get("code")
            if message:
                return str(message)
        if isinstance(error, str):
            return error
        return f"Qwen Omni error: {event}"

    @staticmethod
    def _is_active_response_error(message: str) -> bool:
        return "Conversation already has an active response" in message

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes"}:
                return True
            if normalized in {"false", "0", "no"}:
                return False
        return None

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def shutdown(self) -> None:
        if self._active_ws is not None:
            await self._active_ws.close()
            self._active_ws = None


@dataclass
class _QwenDeferredResponse:
    response_payload: dict[str, Any]
    item_payload: dict[str, Any] | None = None
    item_sent: bool = False


class _QwenResponseCoordinator:
    def __init__(self) -> None:
        self._pending: deque[_QwenDeferredResponse] = deque()
        self._pending_condition = asyncio.Condition()
        self._state_condition = asyncio.Condition()
        self._idle = True
        self._closed = False
        self._current_response: _QwenDeferredResponse | None = None

    async def enqueue(self, request: _QwenDeferredResponse) -> None:
        async with self._pending_condition:
            if self._closed:
                return
            self._pending.append(request)
            self._pending_condition.notify()

    async def _prepend(self, request: _QwenDeferredResponse) -> None:
        async with self._pending_condition:
            if self._closed:
                return
            self._pending.appendleft(request)
            self._pending_condition.notify()

    async def next_request(self) -> _QwenDeferredResponse | None:
        async with self._pending_condition:
            while not self._pending and not self._closed:
                await self._pending_condition.wait()
            if self._pending:
                return self._pending.popleft()
            return None

    async def wait_idle(self) -> bool:
        async with self._state_condition:
            while not self._idle and not self._closed:
                await self._state_condition.wait()
            return self._idle

    async def begin_client_response(self, request: _QwenDeferredResponse) -> None:
        async with self._state_condition:
            self._idle = False
            self._current_response = request
            self._state_condition.notify_all()

    async def release_client_response(self, request: _QwenDeferredResponse) -> None:
        async with self._state_condition:
            if self._current_response is request:
                self._current_response = None
            self._idle = True
            self._state_condition.notify_all()

    async def mark_response_started(self) -> None:
        async with self._state_condition:
            self._idle = False
            self._state_condition.notify_all()

    async def mark_response_done(self) -> None:
        async with self._state_condition:
            self._idle = True
            self._current_response = None
            self._state_condition.notify_all()
        async with self._pending_condition:
            self._pending_condition.notify_all()

    async def mark_active_response_error(self) -> None:
        retry: _QwenDeferredResponse | None = None
        async with self._state_condition:
            if self._current_response is not None:
                retry = self._current_response
                retry.item_sent = True
                self._current_response = None
            self._idle = False
            self._state_condition.notify_all()
        if retry is not None:
            await self._prepend(retry)

    async def close(self) -> None:
        async with self._pending_condition:
            self._closed = True
            self._pending_condition.notify_all()
        async with self._state_condition:
            self._idle = True
            self._state_condition.notify_all()

    async def wait_all_responses_done(self, timeout: float) -> None:
        async def _wait() -> None:
            while True:
                async with self._pending_condition:
                    has_pending = bool(self._pending)
                    closed = self._closed
                async with self._state_condition:
                    idle = self._idle and self._current_response is None
                if closed or (not has_pending and idle):
                    return
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(_wait(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("qwen_omni timed out waiting for deferred response completion")


class _QwenTurnState:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.turn_index = 0
        self.question_id = ""
        self.reply_id = ""
        self.assistant_text = ""
        self.has_audio = False

    @property
    def has_content(self) -> bool:
        return self.has_audio or bool(self.assistant_text)

    def ensure_turn(self) -> None:
        if not self.question_id:
            self.start_next_turn()

    def start_next_turn(self) -> None:
        self.turn_index += 1
        self.question_id = f"{self.session_id}_q{self.turn_index}"
        self.reply_id = f"{self.session_id}_r{self.turn_index}"
        self.assistant_text = ""
        self.has_audio = False

    def reset(self) -> None:
        self.question_id = ""
        self.reply_id = ""
        self.assistant_text = ""
        self.has_audio = False
