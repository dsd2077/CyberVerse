import logging
from typing import AsyncIterator

import numpy as np

from inference.core.types import AudioChunk, PluginConfig
from inference.plugins.tts.base import AudioRechunker, TTSPlugin

logger = logging.getLogger(__name__)


class OpenAITTSPlugin(TTSPlugin):
    name = "tts.openai"

    def __init__(self) -> None:
        self.client = None
        self.voice = "nova"
        self.model = "tts-1"
        self.rechunker = AudioRechunker()
        self._openai_sample_rate = 24000

    async def initialize(self, config: PluginConfig) -> None:
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(api_key=config.params.get("api_key"))
        self.voice = config.params.get("voice", "nova")
        self.model = config.params.get("model", "tts-1")
        self.rechunker = AudioRechunker(
            chunk_samples=17920,
            sample_rate=16000,
        )

    async def synthesize_stream(
        self, text_stream: AsyncIterator[str]
    ) -> AsyncIterator[AudioChunk]:
        async for sentence in text_stream:
            if not sentence.strip():
                continue

            try:
                response = await self.client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=sentence,
                    response_format="pcm",
                )
            except Exception:
                logger.exception("OpenAI TTS API call failed for: %s", sentence[:50])
                continue

            audio_bytes = response.content
            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            )

            if self._openai_sample_rate != 16000:
                audio_np = self._resample(audio_np, self._openai_sample_rate, 16000)

            chunks = self.rechunker.feed(audio_np)
            for chunk in chunks:
                yield chunk

        final_chunk = self.rechunker.flush()
        if final_chunk:
            yield final_chunk

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio with proper anti-aliasing via polyphase filtering."""
        if orig_sr == target_sr:
            return audio
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)

    async def shutdown(self) -> None:
        self.client = None
        self.rechunker.reset()
