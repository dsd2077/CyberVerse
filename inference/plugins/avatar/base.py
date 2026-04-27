from abc import abstractmethod
from typing import AsyncIterator

from inference.core.types import AudioChunk, VideoChunk
from inference.plugins.base import CyberVersePlugin


class AvatarPlugin(CyberVersePlugin):
    @abstractmethod
    async def set_avatar(self, image_path: str, use_face_crop: bool = False) -> None:
        ...

    @abstractmethod
    async def generate_stream(
        self, audio_stream: AsyncIterator[AudioChunk]
    ) -> AsyncIterator[VideoChunk]:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...

    @abstractmethod
    def get_fps(self) -> int:
        ...

    @abstractmethod
    def get_output_dimensions(self) -> tuple[int, int]:
        ...
