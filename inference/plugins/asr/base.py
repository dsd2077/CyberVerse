from abc import abstractmethod
from typing import AsyncIterator

from inference.core.types import ASRRequestConfig, TranscriptEvent
from inference.plugins.base import CyberVersePlugin


class ASRPlugin(CyberVersePlugin):
    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        request_config: ASRRequestConfig | None = None,
    ) -> AsyncIterator[TranscriptEvent]:
        ...
