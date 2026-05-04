from abc import abstractmethod
from typing import AsyncIterator

from inference.core.types import LLMResponseChunk
from inference.plugins.base import CyberVersePlugin


class LLMPlugin(CyberVersePlugin):
    supports_images = False

    @abstractmethod
    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMResponseChunk]:
        ...
