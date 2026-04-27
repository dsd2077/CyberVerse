import tempfile

import grpc

from inference.core.registry import PluginRegistry
from inference.core.types import AudioChunk
from inference.generated import avatar_pb2, avatar_pb2_grpc, common_pb2
from inference.plugins.avatar.base import AvatarPlugin


class AvatarGRPCService(avatar_pb2_grpc.AvatarServiceServicer):

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def _get_plugin(self) -> AvatarPlugin:
        plugin = self.registry.get_by_category("avatar")
        if plugin is None:
            raise RuntimeError("No avatar plugin initialized")
        return plugin

    async def SetAvatar(self, request, context):
        try:
            plugin = self._get_plugin()
            suffix = f".{request.image_format}" if request.image_format else ".png"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(request.image_data)
                image_path = f.name
            await plugin.set_avatar(image_path, request.use_face_crop)
            return avatar_pb2.SetAvatarResponse(success=True, message="Avatar set")
        except Exception as e:
            return avatar_pb2.SetAvatarResponse(success=False, message=str(e))

    async def GenerateStream(self, request_iterator, context):
        plugin = self._get_plugin()

        async def audio_stream():
            async for chunk in request_iterator:
                yield AudioChunk(
                    data=chunk.data,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    format=chunk.format,
                    is_final=chunk.is_final,
                    timestamp_ms=chunk.timestamp_ms,
                )

        async for video_chunk in plugin.generate_stream(audio_stream()):
            yield common_pb2.VideoChunk(
                data=video_chunk.frames.tobytes(),
                width=video_chunk.frames.shape[2],
                height=video_chunk.frames.shape[1],
                num_frames=video_chunk.frames.shape[0],
                fps=video_chunk.fps,
                chunk_index=video_chunk.chunk_index,
                is_final=video_chunk.is_final,
            )

    async def Reset(self, request, context):
        try:
            plugin = self._get_plugin()
            await plugin.reset()
            return avatar_pb2.ResetResponse(success=True)
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return avatar_pb2.ResetResponse(success=False)

    async def GetInfo(self, request, context):
        plugin = self._get_plugin()
        output_width, output_height = plugin.get_output_dimensions()
        return avatar_pb2.AvatarInfo(
            model_name=plugin.name,
            output_fps=plugin.get_fps(),
            output_width=output_width,
            output_height=output_height,
            frames_per_chunk=28,
            chunk_duration_s=1.12,
        )
