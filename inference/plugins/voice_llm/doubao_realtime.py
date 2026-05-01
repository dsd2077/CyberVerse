import asyncio
import json
import logging
import uuid
from typing import AsyncIterator

from inference.core.types import (
    AudioChunk,
    PluginConfig,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.base import VoiceCheckError, VoiceLLMPlugin
from inference.plugins.voice_llm.doubao_config import DoubaoSessionConfig
from inference.plugins.voice_llm.doubao_protocol import (
    DecodedFrame,
    DoubaoEvent,
    MSGTYPE_AUDIO_ONLY_CLIENT,
    MSGTYPE_FULL_CLIENT,
    SERIALIZATION_JSON,
    SERIALIZATION_RAW,
    compress_payload,
    decode_frame,
    decompress_payload,
    encode_frame,
)

logger = logging.getLogger(__name__)
_MAX_OUTPUT_QUEUE = 64


class DoubaoRealtimePlugin(VoiceLLMPlugin):
    """Doubao realtime voice LLM plugin (WebSocket binary protocol)."""

    name = "voice_llm.doubao"

    def __init__(self) -> None:
        self._config: DoubaoSessionConfig | None = None
        self._ws = None
        self._session_id: str | None = None
        self._interrupting = False
        self._dialog_ids: dict[str, str] = {}

    async def initialize(self, config: PluginConfig) -> None:
        self._config = DoubaoSessionConfig.from_plugin_config(config)

    def _effective_config(
        self,
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> DoubaoSessionConfig:
        assert self._config is not None
        if session_config is None:
            return self._config
        return self._config.with_overrides(session_config)

    @staticmethod
    def _decode_payload_text(decoded: DecodedFrame) -> str:
        if not decoded.payload:
            return ""
        try:
            payload = decompress_payload(decoded.payload, decoded.compression_bits)
        except Exception:
            payload = decoded.payload
        if isinstance(payload, (bytes, bytearray)):
            return payload.decode("utf-8", errors="ignore")
        return str(payload)

    async def _recv_expected_control_event(
        self,
        ws,
        *,
        expected_event: int,
        stage: str,
        preserve_provider_error: bool = False,
    ) -> DecodedFrame:
        frame = await ws.recv()
        if isinstance(frame, str):
            raise RuntimeError(f"Doubao {stage} returned text frame unexpectedly")

        decoded = decode_frame(frame)
        payload_text = self._decode_payload_text(decoded)

        if decoded.is_error():
            if preserve_provider_error:
                raise VoiceCheckError(payload_text or f"Doubao {stage} failed")
            message = (
                f"Doubao {stage} failed: code={decoded.error_code} payload={payload_text}"
            )
            logger.error(message)
            raise RuntimeError(message)
        if decoded.is_full_server() and decoded.event == DoubaoEvent.SESSION_FAILED:
            if preserve_provider_error:
                raise VoiceCheckError(payload_text or f"Doubao {stage} failed")
            message = (
                f"Doubao {stage} failed: event={decoded.event} payload={payload_text}"
            )
            logger.error(message)
            raise RuntimeError(message)
        if not decoded.is_full_server():
            message = (
                f"Doubao {stage} returned unexpected frame type={decoded.msg_type_bits}"
            )
            logger.error(message)
            raise RuntimeError(message)
        if decoded.event != expected_event:
            message = (
                f"Doubao {stage} returned unexpected event={decoded.event}, "
                f"expected={expected_event}, payload={payload_text}"
            )
            logger.error(message)
            raise RuntimeError(message)
        return decoded

    async def _send_full_client_event(
        self,
        ws,
        *,
        event: int,
        session_id: str | None,
        config: DoubaoSessionConfig,
        payload: dict | bytes,
    ) -> None:
        if isinstance(payload, (bytes, bytearray)):
            payload_bytes = bytes(payload)
        else:
            payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await ws.send(
            encode_frame(
                msg_type_bits=MSGTYPE_FULL_CLIENT,
                serialization_bits=SERIALIZATION_JSON,
                event=event,
                session_id=session_id,
                payload=compress_payload(payload_bytes, config.compression_bits),
                compression_bits=config.compression_bits,
            )
        )

    async def _start_session(
        self,
        ws,
        *,
        session_id: str,
        config: DoubaoSessionConfig,
        preserve_provider_error: bool = False,
    ) -> str:
        # 1) StartConnection (event=1)
        await self._send_full_client_event(
            ws,
            event=DoubaoEvent.START_CONNECTION,
            session_id=None,
            config=config,
            payload=b"{}",
        )
        # 2) Wait ConnectionStarted (event=50)
        await self._recv_expected_control_event(
            ws,
            expected_event=DoubaoEvent.CONNECTION_STARTED,
            stage="connection handshake",
            preserve_provider_error=preserve_provider_error,
        )

        dialog_id = self._dialog_ids.get(config.conversation_id, "")
        start_session_payload = config.build_start_session_payload(
            dialog_id=dialog_id or None
        )
        speaker = start_session_payload["tts"]["speaker"]
        # 3) StartSession (event=100)
        await self._send_full_client_event(
            ws,
            event=DoubaoEvent.START_SESSION,
            session_id=session_id,
            config=config,
            payload=start_session_payload,
        )
        # 4) Wait SessionStarted (event=150)
        started = await self._recv_expected_control_event(
            ws,
            expected_event=DoubaoEvent.SESSION_STARTED,
            stage=f"start session for speaker={speaker!r}",
            preserve_provider_error=preserve_provider_error,
        )
        try:
            started_payload = decompress_payload(
                started.payload, started.compression_bits
            )
            started_data = json.loads(started_payload)
        except (json.JSONDecodeError, Exception):
            started_data = {}
        dialog_id = str(started_data.get("dialog_id", "") or "")
        if dialog_id and config.conversation_id:
            self._dialog_ids[config.conversation_id] = dialog_id
        return speaker

    async def _finish_session(
        self,
        ws,
        *,
        session_id: str,
        config: DoubaoSessionConfig,
        stage: str,
        preserve_provider_error: bool = False,
    ) -> None:
        # 1) FinishSession (event=102)
        await self._send_full_client_event(
            ws,
            event=DoubaoEvent.FINISH_SESSION,
            session_id=session_id,
            config=config,
            payload=b"{}",
        )
        # 2) Wait SessionFinished (event=152)
        await self._recv_expected_control_event(
            ws,
            expected_event=DoubaoEvent.SESSION_FINISHED,
            stage=stage,
            preserve_provider_error=preserve_provider_error,
        )

    async def check_voice(
        self,
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> None:
        import websockets

        effective_config = self._effective_config(session_config)
        session_id = str(uuid.uuid4())
        connect_id = str(uuid.uuid4())
        headers = effective_config.build_ws_headers(connect_id)

        async with websockets.connect(
            effective_config.ws_url, additional_headers=headers
        ) as ws:
            speaker = await self._start_session(
                ws,
                session_id=session_id,
                config=effective_config,
                preserve_provider_error=True,
            )
            await self._finish_session(
                ws,
                session_id=session_id,
                config=effective_config,
                stage=f"finish session for speaker={speaker!r}",
                preserve_provider_error=True,
            )

    async def converse_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        import websockets

        effective_config = self._effective_config(session_config)

        attempt = 0
        last_error = None
        while attempt <= effective_config.max_retries:
            try:
                async for event in self._converse_stream_inner(input_stream, effective_config):
                    yield event
                return
            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                attempt += 1
                last_error = e
                if attempt > effective_config.max_retries:
                    break
                backoff = min(
                    effective_config.retry_backoff_base * (2 ** (attempt - 1)),
                    effective_config.retry_backoff_max,
                )
                logger.warning(
                    "Doubao connection failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    effective_config.max_retries,
                    backoff,
                    e,
                )
                await asyncio.sleep(backoff)
        raise RuntimeError(
            f"Doubao connection failed after {attempt} attempts: {last_error}"
        )

    async def _converse_stream_inner(
        self, input_stream: AsyncIterator[VoiceLLMInputEvent], config: DoubaoSessionConfig
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        import websockets

        output_queue: asyncio.Queue[VoiceLLMOutputEvent | None] = asyncio.Queue(
            maxsize=_MAX_OUTPUT_QUEUE
        )
        done = asyncio.Event()

        session_id = str(uuid.uuid4())
        connect_id = str(uuid.uuid4())

        headers = config.build_ws_headers(connect_id)

        async with websockets.connect(
            config.ws_url, additional_headers=headers
        ) as ws:
            self._ws = ws
            self._session_id = session_id

            # StartConnection + ConnectionStarted + StartSession + SessionStarted.
            await self._start_session(
                ws,
                session_id=session_id,
                config=config,
            )

            # 5) SayHello (event=300) only when the character explicitly defines one.
            if config.has_welcome_message:
                say_hello_payload = config.build_say_hello_payload()
                await self._send_full_client_event(
                    ws,
                    event=DoubaoEvent.SAY_HELLO,
                    session_id=session_id,
                    config=config,
                    payload=say_hello_payload,
                )

            sender_task = asyncio.create_task(
                self._send_inputs(ws, input_stream, session_id, config)
            )
            receiver_task = asyncio.create_task(
                self._receive_audio(ws, output_queue, done, config, session_id)
            )

            def _on_task_done(task: asyncio.Task) -> None:
                if task.cancelled():
                    return
                exc = task.exception()
                if exc is not None:
                    logger.error("Doubao task failed: %s", exc)
                    done.set()
                    try:
                        output_queue.put_nowait(None)
                    except asyncio.QueueFull:
                        pass

            sender_task.add_done_callback(_on_task_done)
            receiver_task.add_done_callback(_on_task_done)

            try:
                while True:
                    try:
                        event = await asyncio.wait_for(output_queue.get(), timeout=1.0)
                    except asyncio.TimeoutError:
                        if done.is_set():
                            break
                        continue
                    if event is None:
                        break
                    yield event
            finally:
                for task in (sender_task, receiver_task):
                    task.cancel()
                for task in (sender_task, receiver_task):
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass

    async def _send_inputs(
        self, ws, input_stream: AsyncIterator[VoiceLLMInputEvent], session_id: str,
        config: DoubaoSessionConfig,
    ) -> None:
        try:
            sent_text_query = False
            async for event in input_stream:
                if event.text:
                    sent_text_query = True
                    payload = json.dumps(
                        {"content": event.text}, ensure_ascii=False
                    ).encode("utf-8")
                    await ws.send(
                        encode_frame(
                            msg_type_bits=MSGTYPE_FULL_CLIENT,
                            serialization_bits=SERIALIZATION_JSON,
                            event=DoubaoEvent.CHAT_TEXT_QUERY,
                            session_id=session_id,
                            payload=compress_payload(
                                payload, config.compression_bits
                            ),
                            compression_bits=config.compression_bits,
                        )
                    )
                    continue
                chunk_bytes = event.audio
                if not chunk_bytes:
                    continue
                await ws.send(
                    encode_frame(
                        msg_type_bits=MSGTYPE_AUDIO_ONLY_CLIENT,
                        serialization_bits=SERIALIZATION_RAW,
                        event=DoubaoEvent.TASK_REQUEST,
                        session_id=session_id,
                        payload=compress_payload(
                            chunk_bytes, config.compression_bits
                        ),
                        compression_bits=config.compression_bits,
                    )
                )
            # For text mode, wait for REPLY_DONE from server side first; sending
            # FINISH_SESSION immediately can terminate before the reply arrives.
            if not sent_text_query:
                await ws.send(
                    encode_frame(
                        msg_type_bits=MSGTYPE_FULL_CLIENT,
                        serialization_bits=SERIALIZATION_JSON,
                        event=DoubaoEvent.FINISH_SESSION,
                        session_id=session_id,
                        payload=compress_payload(
                            b"{}", config.compression_bits
                        ),
                        compression_bits=config.compression_bits,
                    )
                )
        except Exception:
            logger.exception("Failed to send audio to Doubao")
            raise

    async def _receive_audio(
        self,
        ws,
        output_queue: asyncio.Queue[VoiceLLMOutputEvent | None],
        done: asyncio.Event,
        config: DoubaoSessionConfig,
        session_id: str,
    ) -> None:
        turn_has_audio = False
        turn_final_sent = False
        turn_transcript = ""
        turn_question_id = ""
        turn_reply_id = ""
        last_was_idle_timeout = False
        try:
            async for message in ws:
                if isinstance(message, str):
                    continue
                frame = message
                try:
                    decoded = decode_frame(frame)
                except Exception:
                    logger.warning("Failed to decode Doubao frame (%d bytes)", len(frame))
                    continue

                if decoded.is_audio():
                    audio_payload = decompress_payload(
                        decoded.payload, decoded.compression_bits
                    )
                    logger.debug(
                        "Doubao recv: audio frame, event=%s, %d bytes",
                        decoded.event,
                        len(audio_payload),
                    )
                    await output_queue.put(
                        VoiceLLMOutputEvent(
                            audio=AudioChunk(
                                data=audio_payload,
                                sample_rate=config.output_sample_rate,
                                channels=1,
                                format=config.output_audio_format,
                                is_final=False,
                            ),
                            question_id=turn_question_id,
                            reply_id=turn_reply_id,
                        )
                    )
                    if len(audio_payload) > 0:
                        turn_has_audio = True
                        turn_final_sent = False
                elif decoded.is_full_server():
                    try:
                        text_payload = decompress_payload(
                            decoded.payload, decoded.compression_bits
                        )
                        data = json.loads(text_payload)
                    except (json.JSONDecodeError, Exception):
                        data = {}
                    logger.debug(
                        "Doubao recv: FullServer event=%s payload=%s", decoded.event, data
                    )
                    # Extract transcript from relevant events:
                    # - 351 (TTS_SENTENCE_DONE): 'text' = assistant sentence
                    # - 451 (ASR_RESULT): 'results[0].text' = user speech
                    # - 550 (LLM_TOKEN): 'content' = LLM streaming token
                    assistant_text = ""
                    user_text = ""
                    question_id = str(data.get("question_id", "") or "")
                    reply_id = str(data.get("reply_id", "") or "")

                    if question_id:
                        turn_question_id = question_id
                    if reply_id:
                        turn_reply_id = reply_id

                    if decoded.event == DoubaoEvent.ASR_START:
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                barge_in=True,
                                question_id=turn_question_id,
                                reply_id=turn_reply_id,
                            )
                        )
                    elif decoded.event == DoubaoEvent.TTS_SENTENCE_DONE:
                        assistant_text = data.get("text", "")
                    elif decoded.event == DoubaoEvent.ASR_RESULT:
                        results = data.get("results", [])
                        if results:
                            user_text = results[0].get("text", "")
                            is_interim = results[0].get("is_interim", True)
                            if user_text and not is_interim:
                                await output_queue.put(
                                    VoiceLLMOutputEvent(
                                        user_transcript=user_text,
                                        question_id=turn_question_id,
                                        reply_id=turn_reply_id,
                                    )
                                )
                    elif decoded.event == DoubaoEvent.LLM_TOKEN:
                        assistant_text = data.get("content", "")

                    # LLM tokens provide incremental text for the happy path.
                    # When Doubao only returns sentence-done text with no audio
                    # frames, keep that text as the turn transcript so the Go
                    # side can fall back to local TTS.
                    if assistant_text and decoded.event == DoubaoEvent.LLM_TOKEN:
                        turn_transcript += assistant_text
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                transcript=assistant_text,
                                question_id=turn_question_id,
                                reply_id=turn_reply_id,
                            )
                        )
                    elif (
                        assistant_text
                        and decoded.event == DoubaoEvent.TTS_SENTENCE_DONE
                        and not turn_transcript
                    ):
                        turn_transcript = assistant_text

                    # event 359 (REPLY_DONE) = assistant reply audio fully sent
                    if decoded.event == DoubaoEvent.REPLY_DONE:
                        if turn_has_audio and not turn_final_sent:
                            logger.debug(
                                "Doubao reply done (event=359), emit turn_final marker"
                            )
                            await output_queue.put(
                                VoiceLLMOutputEvent(
                                    audio=AudioChunk(
                                        data=b"",
                                        sample_rate=config.output_sample_rate,
                                        channels=1,
                                        format=config.output_audio_format,
                                        is_final=True,
                                    ),
                                    transcript=turn_transcript,
                                    is_final=True,
                                    question_id=turn_question_id,
                                    reply_id=turn_reply_id,
                                )
                            )
                            turn_final_sent = True
                            turn_transcript = ""
                        elif turn_transcript and not turn_final_sent:
                            logger.debug(
                                "Doubao reply done without audio, emit transcript-only final"
                            )
                            await output_queue.put(
                                VoiceLLMOutputEvent(
                                    transcript=turn_transcript,
                                    is_final=True,
                                    question_id=turn_question_id,
                                    reply_id=turn_reply_id,
                                )
                            )
                            turn_final_sent = True
                            turn_transcript = ""
                        turn_question_id = ""
                        turn_reply_id = ""
                        if config.input_mod == "text":
                            await output_queue.put(None)
                            break
                    elif decoded.event in (
                        DoubaoEvent.SESSION_FINISHED,
                        DoubaoEvent.SESSION_FAILED,
                    ):
                        # Handle interrupt: if we initiated the finish, reset and don't terminate
                        if (
                            self._interrupting
                            and decoded.event == DoubaoEvent.SESSION_FINISHED
                        ):
                            self._interrupting = False
                            continue
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                audio=AudioChunk(
                                    data=b"",
                                    sample_rate=config.output_sample_rate,
                                    channels=1,
                                    format=config.output_audio_format,
                                    is_final=True,
                                ),
                                transcript=turn_transcript,
                                is_final=True,
                                question_id=turn_question_id,
                                reply_id=turn_reply_id,
                            )
                        )
                        turn_question_id = ""
                        turn_reply_id = ""
                        await output_queue.put(None)
                        break
                elif decoded.is_error():
                    try:
                        err_text = decompress_payload(decoded.payload, decoded.compression_bits)
                    except Exception:
                        err_text = decoded.payload[:200]
                    err_text_str = (
                        err_text.decode("utf-8", errors="ignore")
                        if isinstance(err_text, (bytes, bytearray))
                        else str(err_text)
                    )
                    is_idle_timeout = "DialogAudioIdleTimeoutError" in err_text_str

                    if is_idle_timeout:
                        if turn_transcript and not turn_final_sent:
                            logger.info(
                                "Doubao idle timeout with transcript-only reply, emit final transcript"
                            )
                            await output_queue.put(
                                VoiceLLMOutputEvent(
                                    transcript=turn_transcript,
                                    is_final=True,
                                    question_id=turn_question_id,
                                    reply_id=turn_reply_id,
                                )
                            )
                        logger.info(
                            "Doubao idle timeout: keep session open for next turn, payload=%s",
                            err_text_str,
                        )
                        turn_has_audio = False
                        turn_final_sent = False
                        turn_transcript = ""
                        turn_question_id = ""
                        turn_reply_id = ""
                        last_was_idle_timeout = True
                        continue

                    if turn_final_sent:
                        # Reply already completed; idle timeout is expected (e.g. welcome greeting
                        # with no user audio). Log at INFO and skip emitting a duplicate final.
                        logger.info(
                            "Doubao post-reply error (expected idle timeout): code=%s payload=%s",
                            decoded.error_code,
                            err_text,
                        )
                    else:
                        logger.error(
                            "Doubao recv: Error code=%s payload=%s",
                            decoded.error_code, err_text,
                        )
                        await output_queue.put(
                            VoiceLLMOutputEvent(
                                audio=AudioChunk(
                                    data=b"",
                                    sample_rate=config.output_sample_rate,
                                    channels=1,
                                    format=config.output_audio_format,
                                    is_final=True,
                                ),
                                is_final=True,
                                question_id=turn_question_id,
                                reply_id=turn_reply_id,
                            )
                        )
                    await output_queue.put(None)
                    break
        except Exception as exc:
            import websockets
            if isinstance(exc, websockets.ConnectionClosedError) and last_was_idle_timeout:
                logger.info(
                    "Doubao WebSocket closed after idle timeout (expected), ending stream gracefully"
                )
            else:
                logger.exception("Failed to receive audio from Doubao")
                raise
        finally:
            done.set()
            try:
                output_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def interrupt(self) -> None:
        ws = self._ws
        session_id = self._session_id
        if ws is None or session_id is None:
            return
        self._interrupting = True
        try:
            await ws.send(
                encode_frame(
                    msg_type_bits=MSGTYPE_FULL_CLIENT,
                    serialization_bits=SERIALIZATION_JSON,
                    event=DoubaoEvent.FINISH_SESSION,
                    session_id=session_id,
                    payload=compress_payload(
                        b"{}", self._config.compression_bits
                    ),
                    compression_bits=self._config.compression_bits,
                )
            )
        except Exception:
            logger.warning("Failed to send interrupt frame to Doubao")

    async def shutdown(self) -> None:
        return
