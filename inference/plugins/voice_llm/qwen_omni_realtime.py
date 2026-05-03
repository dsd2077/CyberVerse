import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator

from inference.core.types import (
    AudioChunk,
    PluginConfig,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.qwen_endpoint import dashscope_realtime_ws_url
from inference.plugins.voice_llm.base import VoiceCheckError, VoiceLLMPlugin

logger = logging.getLogger(__name__)


class QwenOmniRealtimePlugin(VoiceLLMPlugin):
    """DashScope Qwen Omni realtime VoiceLLM plugin."""

    name = "voice_llm.qwen_omni"

    def __init__(self) -> None:
        self.api_key = ""
        self.model = "qwen3.5-omni-flash-realtime"
        self.ws_url = ""
        self.voice = "Tina"
        self.system_prompt = "你是一个友善的数字人助手。"
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
        if session_config and session_config.input_mode == "text":
            raise RuntimeError(
                "Qwen Omni Realtime provider currently supports realtime audio "
                "input only in DashScope client events; text input is not exposed."
            )

        import websockets

        config = session_config or VoiceLLMSessionConfig()
        ws = await self._connect(websockets)
        self._active_ws = ws
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None] = (
            asyncio.Queue()
        )
        sender_task: asyncio.Task | None = None
        receiver_task: asyncio.Task | None = None
        try:
            await self._configure_session(ws, config)
            sender_task = asyncio.create_task(
                self._send_inputs(ws, input_stream, config.session_id, output_queue)
            )
            receiver_task = asyncio.create_task(
                self._receive_events(ws, config.session_id, output_queue)
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
    ) -> None:
        try:
            async for event in input_stream:
                if event.text:
                    raise RuntimeError(
                        "Qwen Omni Realtime provider currently supports realtime "
                        "audio input only in DashScope client events; text input is "
                        "not exposed."
                    )
                if not event.audio:
                    continue
                await self._send_json(
                    ws,
                    {
                        "type": "input_audio_buffer.append",
                        "event_id": self._event_id(session_id, "audio"),
                        "audio": base64.b64encode(event.audio).decode("ascii"),
                    },
                )
        except Exception as exc:
            await output_queue.put(exc)
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    async def _receive_events(
        self,
        ws: Any,
        session_id: str,
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | Exception | None],
    ) -> None:
        turn_state = _QwenTurnState(session_id=session_id or "qwen_omni")
        try:
            async for message in ws:
                event = self._decode_message(message)
                event_type = event.get("type", "")
                if event_type == "error":
                    raise RuntimeError(self._error_message(event))
                if event_type in {"session.created", "session.updated"}:
                    continue

                if event_type == "input_audio_buffer.speech_started":
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
        optional_values: dict[str, Any] = {
            "enable_search": self.enable_search,
            "search_options": self.search_options,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        for key, value in optional_values.items():
            if value is not None:
                payload[key] = value
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
