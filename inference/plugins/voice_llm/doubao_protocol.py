"""
Doubao realtime binary protocol encoder/decoder.

This module implements the binary framing protocol used by Doubao's realtime
WebSocket API. It handles encoding client frames and decoding server frames
according to the protocol specification.
"""

import gzip
import struct
from dataclasses import dataclass
from typing import Any

# Binary protocol constants
VERSION_AND_HEADER_SIZE = 0x11
RESERVED_BYTE = 0x00

# Message type bits (high 4 bits of second byte)
MSGTYPE_FULL_CLIENT = 0x10
MSGTYPE_AUDIO_ONLY_CLIENT = 0x20
MSGTYPE_FULL_SERVER = 0x90
MSGTYPE_AUDIO_ONLY_SERVER = 0xB0
MSGTYPE_ERROR = 0xF0

# Message flag bits (low 4 bits of second byte)
MSGTYPE_FLAG_WITH_EVENT = 0x04

# Serialization bits (high 4 bits of third byte)
SERIALIZATION_RAW = 0x00
SERIALIZATION_JSON = 0x10

# Compression bits (low 4 bits of third byte)
COMPRESSION_NONE = 0x00
COMPRESSION_GZIP = 0x01


class DoubaoEvent:
    """Event codes used in the Doubao protocol."""

    START_CONNECTION = 1
    CONNECTION_STARTED = 50
    START_SESSION = 100
    SESSION_STARTED = 150
    FINISH_SESSION = 102
    SESSION_FINISHED = 152
    SESSION_FAILED = 153
    TASK_REQUEST = 200
    SAY_HELLO = 300
    CHAT_TEXT_QUERY = 501
    CHAT_TEXT_QUERY_CONFIRMED = 553
    REPLY_START = 350          # 助手开始回复
    TTS_SENTENCE_DONE = 351    # TTS 一句话合成完毕 (含 text 字段)
    AUDIO_DATA = 352           # 音频帧
    REPLY_DONE = 359           # 助手一轮回复结束 (音频全部发完)
    ASR_START = 450            # 用户语音识别开始
    ASR_RESULT = 451           # ASR 中间/最终结果 (含 results[].text)
    TURN_FINISHED = 459        # 用户这轮说完了 (在助手回复之前触发)
    LLM_TOKEN = 550            # LLM 流式 token (含 content 字段)
    LLM_DONE = 559             # LLM 生成完毕


@dataclass
class DecodedFrame:
    """Represents a decoded Doubao protocol frame."""

    msg_type_bits: int
    msg_flags: int
    serialization_bits: int
    compression_bits: int
    event: int | None
    session_id: str | None
    connect_id: str | None
    error_code: int | None
    payload: bytes

    def is_audio(self) -> bool:
        """Returns True if this is an audio-only frame."""
        return self.msg_type_bits == MSGTYPE_AUDIO_ONLY_SERVER

    def is_full_server(self) -> bool:
        """Returns True if this is a full server frame."""
        return self.msg_type_bits == MSGTYPE_FULL_SERVER

    def is_error(self) -> bool:
        """Returns True if this is an error frame."""
        return self.msg_type_bits == MSGTYPE_ERROR


def encode_frame(
    *,
    msg_type_bits: int,
    serialization_bits: int,
    event: int,
    session_id: str | None,
    connect_id: str | None = None,
    payload: bytes,
    compression_bits: int = COMPRESSION_NONE,
) -> bytes:
    """
    Encode a single WebSocket binary frame using Doubao realtime binary protocol.

    This is a minimal encoder for the message subset we need:
    - client messages with `WithEvent` flag
    - raw audio chunks (SerializationRaw)
    - json control frames (SerializationJSON)

    Args:
        msg_type_bits: Message type (e.g., MSGTYPE_FULL_CLIENT, MSGTYPE_AUDIO_ONLY_CLIENT)
        serialization_bits: Serialization format (SERIALIZATION_RAW or SERIALIZATION_JSON)
        event: Event code (from DoubaoEvent)
        session_id: Session identifier (required for most events)
        connect_id: Connection identifier (optional, used for specific events)
        payload: Binary payload data
        compression_bits: Compression type (COMPRESSION_NONE or COMPRESSION_GZIP)

    Returns:
        Encoded binary frame ready to send via WebSocket
    """
    header = bytearray(
        [
            VERSION_AND_HEADER_SIZE,
            msg_type_bits | MSGTYPE_FLAG_WITH_EVENT,
            serialization_bits | compression_bits,
            RESERVED_BYTE,
        ]
    )

    # Event always exists because we always set WithEvent for our outgoing messages.
    header += struct.pack(">i", int(event))

    # protocol.go: writeSessionID() skips session_id for events {1,2,50,51,52}
    if event not in (1, 2, 50, 51, 52):
        if not session_id:
            raise ValueError(f"session_id is required for event={event}")
        sid = session_id.encode("utf-8")
        header += struct.pack(">I", len(sid))
        header += sid

    # protocol.go: readConnectID() reads connect_id for events {50,51,52}.
    # For our outgoing client frames we don't include connect_id, but tests may craft
    # server-return frames for these events.
    if event in (50, 51, 52):
        cid = (connect_id or "").encode("utf-8")
        header += struct.pack(">I", len(cid))
        header += cid

    header += struct.pack(">I", len(payload))
    header += payload
    return bytes(header)


def decode_frame(frame: bytes) -> DecodedFrame:
    """
    Decode a single binary frame into a DecodedFrame.

    Args:
        frame: Binary frame data received from WebSocket

    Returns:
        DecodedFrame object with parsed fields

    Raises:
        ValueError: If frame is malformed or too short
    """
    if len(frame) < 4:
        raise ValueError("frame too short")

    version_and_header_size = frame[0]
    type_and_flag = frame[1]
    serialization_and_compression = frame[2]

    header_size_nibble = version_and_header_size & 0x0F
    header_size_bytes = 4 * header_size_nibble
    offset = 4
    if header_size_bytes > 4:
        if len(frame) < header_size_bytes:
            raise ValueError("frame too short for declared header size")
        offset = header_size_bytes

    msg_type_bits = type_and_flag & 0xF0
    msg_flags = type_and_flag & 0x0F

    contains_event = (msg_flags & MSGTYPE_FLAG_WITH_EVENT) == MSGTYPE_FLAG_WITH_EVENT
    # protocol.go: ContainsSequence() returns true for (PositiveSeq=0b0001) or (NegativeSeq=0b0011)
    contains_sequence = (msg_flags & 0x01) == 0x01 or (msg_flags & 0x03) == 0x03

    payload_len: int
    payload: bytes

    event = None
    session_id = None
    connect_id = None
    error_code = None

    serialization_bits = serialization_and_compression & 0xF0
    compression_bits = serialization_and_compression & 0x0F

    # protocol.go: MsgTypeError readers first read error_code.
    if msg_type_bits == MSGTYPE_ERROR:
        if len(frame) < offset + 4:
            raise ValueError("frame too short for error_code")
        error_code = struct.unpack(">I", frame[offset : offset + 4])[0]
        offset += 4

    # sequence reader for AudioOnlyClient/Server when flag includes it
    if contains_sequence and msg_type_bits in (MSGTYPE_AUDIO_ONLY_CLIENT, MSGTYPE_AUDIO_ONLY_SERVER):
        if len(frame) < offset + 4:
            raise ValueError("frame too short for sequence")
        # not used currently
        _sequence = struct.unpack(">i", frame[offset : offset + 4])[0]
        offset += 4

    if contains_event:
        if len(frame) < offset + 4:
            raise ValueError("frame too short for event")
        event = struct.unpack(">i", frame[offset : offset + 4])[0]
        offset += 4

        # protocol.go: readSessionID() skips session id for events {1,2,50,51,52}
        if event not in (1, 2, 50, 51, 52):
            if len(frame) < offset + 4:
                raise ValueError("frame too short for session_id length")
            sid_len = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            if sid_len:
                if len(frame) < offset + sid_len:
                    raise ValueError("frame too short for session_id")
                session_id = frame[offset : offset + sid_len].decode("utf-8")
                offset += sid_len
            else:
                session_id = ""

        # protocol.go: readConnectID() only for events 50,51,52
        if event in (50, 51, 52):
            if len(frame) < offset + 4:
                raise ValueError("frame too short for connect_id length")
            cid_len = struct.unpack(">I", frame[offset : offset + 4])[0]
            offset += 4
            if cid_len:
                if len(frame) < offset + cid_len:
                    raise ValueError("frame too short for connect_id")
                connect_id = frame[offset : offset + cid_len].decode("utf-8")
                offset += cid_len
            else:
                connect_id = ""

    # payload
    if len(frame) < offset + 4:
        raise ValueError("frame too short for payload length")
    payload_len = struct.unpack(">I", frame[offset : offset + 4])[0]
    offset += 4
    if len(frame) < offset + payload_len:
        raise ValueError("frame too short for payload")
    payload = frame[offset : offset + payload_len]

    return DecodedFrame(
        msg_type_bits=msg_type_bits,
        msg_flags=msg_flags,
        serialization_bits=serialization_bits,
        compression_bits=compression_bits,
        event=event,
        session_id=session_id,
        connect_id=connect_id,
        error_code=error_code,
        payload=payload,
    )


def compress_payload(payload: bytes, compression: int) -> bytes:
    """
    Compress payload data according to compression type.

    Args:
        payload: Binary payload data
        compression: Compression type (COMPRESSION_NONE or COMPRESSION_GZIP)

    Returns:
        Compressed payload (or original if compression is COMPRESSION_NONE)
    """
    if compression == COMPRESSION_GZIP:
        return gzip.compress(payload)
    return payload


def decompress_payload(payload: bytes, compression: int) -> bytes:
    """
    Decompress payload data according to compression type.

    Args:
        payload: Binary payload data (possibly compressed)
        compression: Compression type (COMPRESSION_NONE or COMPRESSION_GZIP)

    Returns:
        Decompressed payload (or original if compression is COMPRESSION_NONE)
    """
    if compression == COMPRESSION_GZIP:
        return gzip.decompress(payload)
    return payload
