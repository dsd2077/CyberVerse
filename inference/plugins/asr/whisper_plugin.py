import asyncio
import logging
from typing import AsyncIterator

import numpy as np

from inference.core.types import ASRRequestConfig, AudioChunk, PluginConfig, TranscriptEvent
from inference.plugins.asr.base import ASRPlugin

logger = logging.getLogger(__name__)


class WhisperASRPlugin(ASRPlugin):
    """OpenAI Whisper-based ASR plugin.

    Accumulates audio chunks, detects silence boundaries via energy threshold,
    and transcribes completed utterances using Whisper.
    """

    name = "asr.whisper"

    def __init__(self) -> None:
        self.model = None
        self.model_size = "base"
        self.language: str | None = None
        self.device = "cpu"
        self._min_audio_seconds = 1.0
        self._silence_threshold = 0.01
        self._silence_duration = 0.5
        self._sample_rate = 16000

    async def initialize(self, config: PluginConfig) -> None:
        self.model_size = config.params.get("model_size", "base")
        self.device = config.params.get("device", "cpu")
        lang = config.params.get("language", "auto")
        self.language = None if lang == "auto" else lang
        self._min_audio_seconds = float(config.params.get("min_audio_seconds", "1.0"))
        self._silence_threshold = float(config.params.get("silence_threshold", "0.01"))
        self._silence_duration = float(config.params.get("silence_duration", "0.5"))

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model)

    def _load_model(self) -> None:
        import whisper

        self.model = whisper.load_model(self.model_size, device=self.device)
        logger.info("Whisper model loaded: size=%s device=%s", self.model_size, self.device)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        request_config: ASRRequestConfig | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        buffer = np.array([], dtype=np.float32)
        silence_samples = 0
        silence_limit = int(self._silence_duration * self._sample_rate)
        min_samples = int(self._min_audio_seconds * self._sample_rate)

        async for chunk_bytes in audio_stream:
            audio_np = np.frombuffer(chunk_bytes, dtype=np.float32)
            buffer = np.concatenate([buffer, audio_np])

            # Track silence
            rms = np.sqrt(np.mean(audio_np**2)) if len(audio_np) > 0 else 0.0
            if rms < self._silence_threshold:
                silence_samples += len(audio_np)
            else:
                silence_samples = 0

            # Transcribe when: enough audio AND silence detected
            if len(buffer) >= min_samples and silence_samples >= silence_limit:
                event = await self._transcribe_buffer(buffer)
                if event and event.text.strip():
                    event.is_final = True
                    yield event
                buffer = np.array([], dtype=np.float32)
                silence_samples = 0

        # Flush remaining audio
        if len(buffer) >= int(0.3 * self._sample_rate):
            event = await self._transcribe_buffer(buffer)
            if event and event.text.strip():
                event.is_final = True
                yield event

    async def _transcribe_buffer(self, audio: np.ndarray) -> TranscriptEvent | None:
        if self.model is None:
            return None
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: np.ndarray) -> TranscriptEvent:
        result = self.model.transcribe(
            audio,
            language=self.language,
            fp16=(self.device != "cpu"),
        )
        text = result.get("text", "").strip()
        language = result.get("language", "")
        segments = result.get("segments", [])
        avg_confidence = 0.0
        if segments:
            probs = [s.get("no_speech_prob", 0.0) for s in segments]
            avg_confidence = 1.0 - (sum(probs) / len(probs))

        return TranscriptEvent(
            text=text,
            is_final=False,
            language=language,
            confidence=avg_confidence,
        )

    async def shutdown(self) -> None:
        self.model = None
        logger.info("Whisper model unloaded")
