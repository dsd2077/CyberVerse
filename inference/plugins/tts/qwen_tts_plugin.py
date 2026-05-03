import base64
import json
import logging
import time
from math import gcd
from typing import Any, AsyncIterator

import numpy as np

from inference.core.types import AudioChunk, PluginConfig, TTSRequestConfig
from inference.plugins.qwen_endpoint import dashscope_realtime_ws_url
from inference.plugins.tts.base import AudioRechunker, TTSPlugin

logger = logging.getLogger(__name__)


class QwenTTSPlugin(TTSPlugin):
    """DashScope Qwen realtime TTS plugin."""

    name = "tts.qwen"

    def __init__(self) -> None:
        self.api_key = ""
        self.model = "qwen3-tts-flash-realtime"
        self.ws_url = ""
        self.voice = "Momo"
        self.sample_rate = 24000
        self.target_sample_rate = 16000
        self.rechunk_samples = 17920

    async def initialize(self, config: PluginConfig) -> None:
        self.api_key = config.params.get("api_key", "")
        self.model = config.params.get("model", self.model)
        self.ws_url = dashscope_realtime_ws_url(self.model, "DASHSCOPE_TTS_WS_URL")
        self.voice = config.params.get("voice", self.voice)
        self.sample_rate = int(config.params.get("sample_rate", self.sample_rate))
        self.target_sample_rate = int(
            config.params.get("target_sample_rate", self.target_sample_rate)
        )
        self.rechunk_samples = int(
            config.params.get("rechunk_samples", self.rechunk_samples)
        )

    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        request_config: TTSRequestConfig | None = None,
    ) -> AsyncIterator[AudioChunk]:
        import websockets

        voice = (request_config.voice if request_config else "") or self.voice
        session_id = (request_config.session_id if request_config else "") or ""
        rechunker = AudioRechunker(
            chunk_samples=self.rechunk_samples,
            sample_rate=self.target_sample_rate,
        )

        ws = await self._connect(websockets)
        try:
            await self._configure_session(ws, voice)

            async for text in text_stream:
                text = text.strip()
                if not text:
                    continue

                await self._send_json(
                    ws,
                    {
                        "type": "input_text_buffer.append",
                        "event_id": self._event_id(session_id),
                        "text": text,
                    },
                )
                await self._send_json(
                    ws,
                    {
                        "type": "input_text_buffer.commit",
                        "event_id": self._event_id(session_id, "commit"),
                    },
                )

                async for audio in self._receive_response_audio(ws, rechunker):
                    yield audio

            final_chunk = rechunker.flush()
            if final_chunk:
                yield final_chunk
        finally:
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

    async def _configure_session(self, ws: Any, voice: str) -> None:
        await self._send_json(
            ws,
            {
                "type": "session.update",
                "event_id": self._event_id("", "session"),
                "session": {
                    "mode": "server_commit",
                    "voice": voice or "Momo",
                    "response_format": "pcm",
                    "sample_rate": self.sample_rate,
                    "channels": 1,
                    "bit_depth": 16,
                },
            },
        )

        while True:
            event = json.loads(await ws.recv())
            event_type = event.get("type", "")
            if event_type in {"session.created", "session.updated"}:
                return
            if event_type == "error":
                raise RuntimeError(f"Qwen TTS session error: {event}")

    async def _receive_response_audio(
        self,
        ws: Any,
        rechunker: AudioRechunker,
    ) -> AsyncIterator[AudioChunk]:
        while True:
            event = json.loads(await ws.recv())
            event_type = event.get("type", "")

            if event_type == "response.audio.delta":
                delta = event.get("delta", "")
                if not delta:
                    continue
                try:
                    pcm = base64.b64decode(delta)
                    audio = (
                        np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    if self.sample_rate != self.target_sample_rate:
                        audio = self._resample(
                            audio,
                            self.sample_rate,
                            self.target_sample_rate,
                        )
                    for chunk in rechunker.feed(audio):
                        yield chunk
                except Exception:
                    logger.exception("Failed to decode Qwen TTS audio delta")
                continue

            if event_type in {"response.done", "response.audio.done", "output.done"}:
                return

            if event_type == "error":
                raise RuntimeError(f"Qwen TTS response error: {event}")

    @staticmethod
    async def _send_json(ws: Any, payload: dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _event_id(session_id: str, suffix: str = "evt") -> str:
        base = session_id or "qwen_tts"
        return f"{base}_{suffix}_{int(time.time() * 1000)}"

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio.astype(np.float32)
        from scipy.signal import resample_poly

        divisor = gcd(orig_sr, target_sr)
        return resample_poly(
            audio,
            target_sr // divisor,
            orig_sr // divisor,
        ).astype(np.float32)

    async def shutdown(self) -> None:
        return None
