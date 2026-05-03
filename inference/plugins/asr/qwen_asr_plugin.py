import asyncio
import base64
import json
import logging
import time
from typing import Any, AsyncIterator

from inference.core.types import ASRRequestConfig, PluginConfig, TranscriptEvent
from inference.plugins.asr.base import ASRPlugin
from inference.plugins.qwen_endpoint import dashscope_realtime_ws_url

logger = logging.getLogger(__name__)


class QwenASRPlugin(ASRPlugin):
    """DashScope Qwen realtime ASR plugin."""

    name = "asr.qwen"

    def __init__(self) -> None:
        self.api_key = ""
        self.model = "qwen3-asr-flash-realtime"
        self.ws_url = ""
        self.language = "auto"
        self.sample_rate = 16000
        self.vad_threshold = 0.5
        self.vad_silence_duration_ms = 1000

    async def initialize(self, config: PluginConfig) -> None:
        self.api_key = config.params.get("api_key", "")
        self.model = config.params.get("model", self.model)
        self.ws_url = dashscope_realtime_ws_url(self.model, "DASHSCOPE_ASR_WS_URL")
        self.language = config.params.get("language", self.language)
        self.sample_rate = int(config.params.get("sample_rate", self.sample_rate))
        self.vad_threshold = float(
            config.params.get("vad_threshold", self.vad_threshold)
        )
        self.vad_silence_duration_ms = int(
            config.params.get(
                "vad_silence_duration_ms", self.vad_silence_duration_ms
            )
        )

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        request_config: ASRRequestConfig | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        import websockets

        language = (request_config.language if request_config else "") or self.language
        session_id = (request_config.session_id if request_config else "") or ""
        transcription_params: dict[str, Any] = {}
        if language and language != "auto":
            transcription_params["language"] = language

        ws = await self._connect(websockets)
        sender_task: asyncio.Task | None = None
        try:
            await self._send_json(
                ws,
                {
                    "type": "session.update",
                    "event_id": self._event_id(session_id, "session"),
                    "session": {
                        "input_audio_format": "pcm",
                        "sample_rate": self.sample_rate,
                        "input_audio_transcription": transcription_params,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self.vad_threshold,
                            "silence_duration_ms": self.vad_silence_duration_ms,
                        },
                    },
                },
            )

            sender_task = asyncio.create_task(
                self._send_audio(ws, audio_stream, session_id)
            )

            async for message in ws:
                event = json.loads(message)
                event_type = event.get("type", "")
                if event_type == "error":
                    raise RuntimeError(f"Qwen ASR error: {event}")

                transcript = self._extract_transcript(event)
                if not transcript:
                    continue

                is_final = self._is_final_event(event)
                yield TranscriptEvent(
                    text=transcript,
                    is_final=is_final,
                    language=event.get(
                        "language", language if language != "auto" else ""
                    ),
                    confidence=float(event.get("confidence", 0.0) or 0.0),
                )
        finally:
            if sender_task and not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
            await ws.close()

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

    async def _send_audio(
        self,
        ws: Any,
        audio_stream: AsyncIterator[bytes],
        session_id: str,
    ) -> None:
        async for chunk in audio_stream:
            if not chunk:
                continue
            await self._send_json(
                ws,
                {
                    "type": "input_audio_buffer.append",
                    "event_id": self._event_id(session_id, "audio"),
                    "audio": base64.b64encode(chunk).decode("ascii"),
                },
            )

        try:
            await self._send_json(
                ws,
                {
                    "type": "session.finish",
                    "event_id": self._event_id(session_id, "finish"),
                },
            )
        except Exception:
            logger.debug(
                "Qwen ASR finish failed after audio stream ended", exc_info=True
            )

    @staticmethod
    async def _send_json(ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _event_id(session_id: str, suffix: str) -> str:
        base = session_id or "qwen_asr"
        return f"{base}_{suffix}_{int(time.time() * 1000)}"

    @classmethod
    def _extract_transcript(cls, event: dict[str, Any]) -> str:
        for key in ("transcript", "stash", "text", "delta"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        for key in ("item", "result", "payload", "output"):
            nested = event.get(key)
            if isinstance(nested, dict):
                text = cls._extract_transcript(nested)
                if text:
                    return text

        choices = event.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    text = cls._extract_transcript(choice)
                    if text:
                        return text

        return ""

    @staticmethod
    def _is_final_event(event: dict[str, Any]) -> bool:
        event_type = str(event.get("type", "")).lower()
        if event.get("is_final") is True or event.get("final") is True:
            return True
        return any(token in event_type for token in ("completed", "final", "done"))

    async def shutdown(self) -> None:
        return None
