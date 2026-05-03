from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class AudioChunk:
    data: bytes
    sample_rate: int = 16000
    channels: int = 1
    format: str = "float32"
    is_final: bool = False
    timestamp_ms: int = 0
    duration_ms: int = 0
    session_id: str = ""
    question_id: str = ""
    reply_id: str = ""
    turn_seq: int = 0
    user_final_unix_ms: int = 0


@dataclass
class VideoChunk:
    frames: np.ndarray  # (N, H, W, 3) uint8
    fps: int = 25
    chunk_index: int = 0
    is_final: bool = False
    trace_generation_started_since_user_final_ms: int = -1


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool = False
    language: str = ""
    confidence: float = 0.0


@dataclass
class LLMResponseChunk:
    token: str
    accumulated_text: str = ""
    is_sentence_end: bool = False
    is_final: bool = False


@dataclass
class ASRRequestConfig:
    provider: str = ""
    language: str = ""
    session_id: str = ""


@dataclass
class TTSRequestConfig:
    provider: str = ""
    voice: str = ""
    speaking_style: str = ""
    language: str = ""
    session_id: str = ""


@dataclass
class VoiceLLMOutputEvent:
    audio: AudioChunk | None = None
    transcript: str = ""
    user_transcript: str = ""
    is_final: bool = False
    question_id: str = ""
    reply_id: str = ""
    barge_in: bool = False


@dataclass
class VoiceLLMInputEvent:
    audio: bytes = b""
    text: str = ""


@dataclass
class VoiceLLMDialogContextItem:
    role: str
    text: str
    timestamp: int = 0


@dataclass
class VoiceLLMSessionConfig:
    """Per-session character config passed from Go through gRPC."""
    session_id: str = ""
    system_prompt: str = ""
    voice: str = ""
    bot_name: str = ""
    speaking_style: str = ""
    welcome_message: str | None = None
    input_mode: str = ""
    dialog_context: list[VoiceLLMDialogContextItem] = field(default_factory=list)


@dataclass
class PluginConfig:
    plugin_name: str
    params: dict[str, Any] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)
