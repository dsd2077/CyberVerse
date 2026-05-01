import tempfile
import logging

import grpc

from inference.core.registry import PluginRegistry
from inference.core.types import AudioChunk
from inference.generated import avatar_pb2, avatar_pb2_grpc, common_pb2
from inference.plugins.avatar.base import AvatarPlugin

logger = logging.getLogger(__name__)


def _metadata_value(context, key: str) -> str:
    for item_key, item_value in context.invocation_metadata():
        if item_key.lower() == key:
            return str(item_value)
    return ""


def _voice_trace_log(
    event: str,
    *,
    session_id: str,
    turn_seq: int,
    reply_id: str,
    question_id: str = "",
    since_user_final_ms: str = "-",
    **fields,
) -> None:
    parts = [
        f"voice_trace event={event:<30}",
        f"sid={session_id or '-'}",
        f"turn={turn_seq}",
        f"reply={reply_id or '-'}",
        f"qid={question_id or '-'}",
        f"since_user_final_ms={since_user_final_ms}",
    ]
    for key, value in fields.items():
        parts.append(f"{key}={value}")
    logger.info(" ".join(parts))


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
        session_id = _metadata_value(context, "x-cyberverse-session-id")
        question_id = _metadata_value(context, "x-cyberverse-question-id")
        reply_id = _metadata_value(context, "x-cyberverse-reply-id")
        try:
            turn_seq = int(_metadata_value(context, "x-cyberverse-turn-seq") or "0")
        except ValueError:
            turn_seq = 0

        _voice_trace_log(
            "avatar_grpc_stream_opened",
            session_id=session_id,
            turn_seq=turn_seq,
            reply_id=reply_id,
            question_id=question_id,
        )

        async def audio_stream():
            first_audio = True
            async for chunk in request_iterator:
                if first_audio:
                    first_audio = False
                    _voice_trace_log(
                        "avatar_grpc_first_audio_received",
                        session_id=session_id,
                        turn_seq=turn_seq,
                        reply_id=reply_id,
                        question_id=question_id,
                    )
                yield AudioChunk(
                    data=chunk.data,
                    sample_rate=chunk.sample_rate,
                    channels=chunk.channels,
                    format=chunk.format,
                    is_final=chunk.is_final,
                    timestamp_ms=chunk.timestamp_ms,
                    session_id=session_id,
                    question_id=question_id,
                    reply_id=reply_id,
                    turn_seq=turn_seq,
                )

        first_video = True
        async for video_chunk in plugin.generate_stream(audio_stream()):
            if first_video:
                first_video = False
                _voice_trace_log(
                    "avatar_grpc_first_video_yielded",
                    session_id=session_id,
                    turn_seq=turn_seq,
                    reply_id=reply_id,
                    question_id=question_id,
                )
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
