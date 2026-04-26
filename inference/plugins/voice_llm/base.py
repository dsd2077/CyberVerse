from abc import abstractmethod
from typing import AsyncIterator

from inference.core.types import VoiceLLMInputEvent, VoiceLLMOutputEvent, VoiceLLMSessionConfig
from inference.plugins.base import CyberVersePlugin


class VoiceCheckError(RuntimeError):
    """Raised when a provider rejects a voice check request."""


class VoiceLLMPlugin(CyberVersePlugin):
    @abstractmethod
    async def check_voice(
        self,
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> None:
        ...

    @abstractmethod
    async def converse_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        ...

    async def interrupt(self) -> None:
        pass
