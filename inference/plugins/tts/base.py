from abc import abstractmethod
from typing import AsyncIterator

import numpy as np

from inference.core.types import AudioChunk, TTSRequestConfig
from inference.plugins.base import CyberVersePlugin


class TTSPlugin(CyberVersePlugin):
    @abstractmethod
    async def synthesize_stream(
        self,
        text_stream: AsyncIterator[str],
        request_config: TTSRequestConfig | None = None,
    ) -> AsyncIterator[AudioChunk]:
        ...


class AudioRechunker:
    """Rechunk variable-length TTS audio into fixed-size chunks aligned with Avatar model.

    FlashHead: frame_num=33, motion_frames=5, effective=28
    chunk_duration = 28/25 = 1.12s = 17920 samples @ 16kHz
    """

    def __init__(self, chunk_samples: int = 17920, sample_rate: int = 16000):
        self.chunk_samples = chunk_samples
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)

    def feed(self, audio: np.ndarray) -> list[AudioChunk]:
        self.buffer = np.concatenate([self.buffer, audio])
        chunks = []
        while len(self.buffer) >= self.chunk_samples:
            chunk_data = self.buffer[: self.chunk_samples]
            self.buffer = self.buffer[self.chunk_samples :]
            chunks.append(
                AudioChunk(
                    data=chunk_data.astype(np.float32).tobytes(),
                    sample_rate=self.sample_rate,
                    duration_ms=int(self.chunk_samples / self.sample_rate * 1000),
                    is_final=False,
                )
            )
        return chunks

    def flush(self) -> AudioChunk | None:
        if len(self.buffer) > 0:
            padded = np.zeros(self.chunk_samples, dtype=np.float32)
            padded[: len(self.buffer)] = self.buffer
            self.buffer = np.array([], dtype=np.float32)
            return AudioChunk(
                data=padded.tobytes(),
                sample_rate=self.sample_rate,
                duration_ms=int(self.chunk_samples / self.sample_rate * 1000),
                is_final=True,
            )
        return None

    def reset(self) -> None:
        self.buffer = np.array([], dtype=np.float32)
