from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import replace
from typing import Any, AsyncIterator

from inference.core.registry import import_plugin_class
from inference.core.types import (
    PluginConfig,
    ToolCall,
    ToolDefinition,
    ToolResult,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.base import VoiceLLMPlugin
from inference.plugins.voice_llm.persona.runtime import LocalTaskRuntime
from inference.plugins.voice_llm.persona.supervisor import PendingSubAgentTask, PersonaSupervisor, SupervisorToolResult

logger = logging.getLogger(__name__)


PERSONA_TOOL_DEFINITIONS = [
    ToolDefinition(
        name="wait_for_more_input",
        description="当用户当前话语尚不完整，需要更多输入后才能回应或执行操作时使用。",
        parameters={
            "type": "object",
            "properties": {
                "partial_text": {
                    "type": "string",
                    "description": "已经听到的部分话语。",
                },
                "reason": {
                    "type": "string",
                    "description": "简要说明意图为何尚不完整。",
                },
            },
            "required": ["partial_text"],
        },
    ),
    ToolDefinition(
        name="create_task",
        description="为搜索、调研、聚合或报告类请求创建 CyberVerse 后台任务。",
        parameters={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "用自然语言描述需要后台 SubAgent 完成的任务。不要拆解工具、类型或执行参数。",
                },
            },
            "required": ["description"],
        },
    ),
    ToolDefinition(
        name="get_task_status",
        description="获取当前会话中最新活跃的 CyberVerse 后台任务状态。",
        parameters={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="cancel_task",
        description="取消当前会话中最新活跃的 CyberVerse 后台任务。",
        parameters={"type": "object", "properties": {}},
    ),
]

PERSONA_AGENT_INSTRUCTIONS = """你是 CyberVerse 数字人 PersonaAgent，直接通过语音和用户对话。
你需要先判断用户当前表达是否已经构成完整意图。
普通寒暄、问答和闲聊：直接自然回答。
语义明显未完成、像半句话、铺垫、犹豫或还在补充：调用 wait_for_more_input。
搜索、查询热点、查询知乎热榜、调研、整理资料、生成报告、生成网页或需要较长后台处理：必须调用 create_task，不要只用口头承诺代替工具调用。
当用户已经给出明确可执行的指令时，不能再追问领域、方向、范围、偏好或“想看哪些方面”；直接调用 create_task 执行。
“看一下今天知乎热榜”“帮我查一下知乎新鲜事”“用知乎搜索一下宇树科技”这类请求已经足够明确，必须直接创建任务。
调用 create_task 时只填写 description，用自然语言描述后台任务；不要决定任务类型、标题或具体工具参数。
调用 create_task 后，最多用一句很短的确认，例如“好的，我去查。”不要做空泛等待播报、不要承诺很快返回结果、不要再追加问题。
询问后台任务进度：调用 get_task_status。
要求取消、停止、不用继续当前后台任务：调用 cancel_task。

"""


class PersonaAgentPlugin(VoiceLLMPlugin):
    """LangGraph-backed persona wrapper for an underlying realtime omni provider.

    The public gRPC wire shape remains the existing VoiceLLM stream. Native tool
    calls are consumed inside this wrapper and are never forwarded to Go or the UI.
    """

    name = "persona.persona"

    def __init__(self) -> None:
        self.model_provider = "doubao"
        self.model_plugin: VoiceLLMPlugin | None = None
        self.task_runtime: LocalTaskRuntime | None = None
        self.supervisor: PersonaSupervisor | None = None
        self.checkpoint_db_path = ""
        self.task_poll_interval_seconds = 1.0
        self.task_monitor_timeout_seconds = 1800.0

    async def initialize(self, config: PluginConfig) -> None:
        self.model_provider = str(config.params.get("model_provider") or "doubao").strip()
        if not self.model_provider or self.model_provider == "persona":
            raise ValueError("persona model_provider must reference a concrete omni provider")

        self.checkpoint_db_path = str(
            config.params.get("checkpoint_db_path")
            or os.getenv("LANGGRAPH_CHECKPOINT_DB")
            or os.path.join(
                os.getenv("CYBERVERSE_CONFIG_DIR", "."),
                "data",
                "tasks",
                "langgraph_checkpoints.db",
            )
        )
        self.task_poll_interval_seconds = max(
            0.1,
            float(config.params.get("task_poll_interval_seconds") or self.task_poll_interval_seconds),
        )
        self.task_monitor_timeout_seconds = max(
            1.0,
            float(config.params.get("task_monitor_timeout_seconds") or self.task_monitor_timeout_seconds),
        )

        omni_config = config.shared.get("omni", {})
        if not isinstance(omni_config, dict):
            raise ValueError("persona provider requires shared omni config")
        provider_conf = omni_config.get(self.model_provider)
        if not isinstance(provider_conf, dict):
            raise ValueError(f"persona model_provider {self.model_provider!r} is not configured")
        class_path = provider_conf.get("plugin_class")
        if not class_path:
            raise ValueError(f"persona model_provider {self.model_provider!r} has no plugin_class")

        plugin_cls = import_plugin_class(str(class_path))
        model_plugin = plugin_cls()
        params = {k: v for k, v in provider_conf.items() if k != "plugin_class"}
        await model_plugin.initialize(
            PluginConfig(
                plugin_name=f"omni.{self.model_provider}",
                params=params,
                shared=config.shared,
            )
        )
        if not isinstance(model_plugin, VoiceLLMPlugin):
            raise TypeError(f"{class_path} is not a VoiceLLMPlugin")
        self.model_plugin = model_plugin

        runtime_config = config.shared.get("runtime_config")
        if not isinstance(runtime_config, dict):
            runtime_config = {
                "inference": {
                    "llm": config.shared.get("llm", {}),
                    "persona_agent": config.params,
                }
            }
        self.task_runtime = LocalTaskRuntime(
            runtime_config=runtime_config,
            max_active_tasks_per_session=int(config.params.get("max_active_tasks_per_session") or 3),
        )
        self.supervisor = PersonaSupervisor(
            runtime=self.task_runtime,
            checkpoint_db_path=self.checkpoint_db_path,
            task_poll_interval_seconds=self.task_poll_interval_seconds,
            task_monitor_timeout_seconds=self.task_monitor_timeout_seconds,
        )
        await self.supervisor.initialize()

    async def shutdown(self) -> None:
        if self.model_plugin is not None:
            await self.model_plugin.shutdown()
        if self.supervisor is not None:
            await self.supervisor.shutdown()
            self.supervisor = None

    async def check_voice(self, session_config: VoiceLLMSessionConfig | None = None) -> None:
        if self.model_plugin is None:
            raise RuntimeError("persona model plugin is not initialized")
        await self.model_plugin.check_voice(session_config)

    async def interrupt(self) -> None:
        if self.model_plugin is not None:
            await self.model_plugin.interrupt()

    async def _execute_tool(self, call: ToolCall, session_id: str) -> SupervisorToolResult:
        if self.supervisor is None:
            raise RuntimeError("persona supervisor is not initialized")
        return await self.supervisor.handle_tool_call(call, session_id)

    @staticmethod
    def _clean_text(text: Any) -> str:
        return str(text or "").strip()

    @staticmethod
    def _needs_space(left: str, right: str) -> bool:
        if not left or not right:
            return False
        return left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum()

    @classmethod
    def _merge_text_segments(cls, segments: list[str]) -> str:
        merged = ""
        for segment in segments:
            text = cls._clean_text(segment)
            if not text:
                continue
            if not merged:
                merged = text
                continue
            if text in merged:
                continue
            if merged in text:
                merged = text
                continue
            separator = " " if cls._needs_space(merged, text) else ""
            merged = f"{merged}{separator}{text}"
        return merged.strip()

    @classmethod
    def _partial_text_for_wait(cls, call: ToolCall, turn_transcripts: list[str]) -> str:
        args = call.arguments or {}
        return cls._clean_text(args.get("partial_text") or args.get("text")) or cls._merge_text_segments(turn_transcripts)

    @classmethod
    def _final_user_text(
        cls,
        call: ToolCall,
        pending_partials: list[str],
        turn_transcripts: list[str],
    ) -> str:
        args = call.arguments or {}
        tool_text = cls._clean_text(
            args.get("description")
            or args.get("user_request")
            or args.get("request")
            or args.get("text")
        )
        current_text = tool_text or cls._merge_text_segments(turn_transcripts)
        return cls._merge_text_segments([*pending_partials, current_text])

    @staticmethod
    def _has_assistant_output(event: VoiceLLMOutputEvent) -> bool:
        return bool(event.transcript or event.audio or event.is_final)

    @staticmethod
    def _clip_text(value: Any, limit: int = 180) -> str:
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @classmethod
    def _tool_calls_for_log(cls, calls: list[ToolCall]) -> list[dict[str, Any]]:
        logged: list[dict[str, Any]] = []
        for call in calls:
            args = call.arguments or {}
            logged.append(
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": cls._clip_text(json.dumps(args, ensure_ascii=False, sort_keys=True)),
                }
            )
        return logged

    @classmethod
    def _model_event_kind(cls, event: VoiceLLMOutputEvent) -> str:
        if event.tool_calls:
            return "tool_call"
        if event.user_transcript:
            return "user_transcript"
        if event.barge_in:
            return "turn_started"
        if event.is_final:
            return "assistant_final"
        if event.transcript:
            return "assistant_delta"
        if event.audio is not None:
            return "audio_delta"
        return "event"

    @classmethod
    def _log_model_event(cls, session_id: str, event: VoiceLLMOutputEvent) -> None:
        kind = cls._model_event_kind(event)
        audio = event.audio
        fields: dict[str, Any] = {
            "question_id": event.question_id,
            "reply_id": event.reply_id,
            "is_final": event.is_final,
            "barge_in": event.barge_in,
        }
        if event.user_transcript:
            fields["user_transcript"] = cls._clip_text(event.user_transcript)
        if event.transcript:
            fields["transcript"] = cls._clip_text(event.transcript)
        if audio is not None:
            fields["audio"] = {
                "bytes": len(audio.data or b""),
                "sample_rate": audio.sample_rate,
                "is_final": audio.is_final,
            }
        if event.tool_calls:
            fields["tool_calls"] = cls._tool_calls_for_log(event.tool_calls)
        info_kinds = {"turn_started", "user_transcript", "tool_call", "assistant_final"}
        log = logger.info if kind in info_kinds else logger.debug
        log(
            "persona model event session=%s kind=%s fields=%s",
            session_id or "",
            kind,
            json.dumps(fields, ensure_ascii=False, sort_keys=True),
        )

    @staticmethod
    def _task_event_payload(task: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": "task_event",
            "task_id": event.get("task_id") or task.get("id"),
            "session_id": task.get("session_id"),
            "seq": event.get("seq"),
            "event_type": event.get("event_type"),
            "status": event.get("status") or task.get("status"),
            "message": event.get("message") or "",
            "progress": event.get("progress", task.get("progress", 0)),
            "created_at": event.get("created_at"),
            "task": task,
        }
        event_payload = event.get("payload")
        if isinstance(event_payload, str):
            try:
                event_payload = json.loads(event_payload)
            except json.JSONDecodeError:
                event_payload = {}
        if isinstance(event_payload, dict) and event_payload:
            payload["payload"] = event_payload
        return payload

    @staticmethod
    def _drain_task_events(queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
        drained: list[dict[str, Any]] = []
        while True:
            try:
                drained.append(queue.get_nowait())
            except asyncio.QueueEmpty:
                return drained

    async def _run_async_task(
        self,
        pending: PendingSubAgentTask,
        injected: asyncio.Queue[VoiceLLMInputEvent],
    ) -> None:
        if self.supervisor is None:
            raise RuntimeError("persona supervisor is not initialized")
        prompt = await self.supervisor.run_pending_task(pending)
        await injected.put(VoiceLLMInputEvent(text=prompt))

    @staticmethod
    def _persona_system_prompt(session_config: VoiceLLMSessionConfig) -> str:
        prompt = (session_config.system_prompt or "").strip()
        if not prompt:
            return PERSONA_AGENT_INSTRUCTIONS
        return f"{PERSONA_AGENT_INSTRUCTIONS}\n\n角色设定：\n{prompt}"

    async def _merged_input_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        injected: asyncio.Queue[VoiceLLMInputEvent],
    ) -> AsyncIterator[VoiceLLMInputEvent]:
        source = input_stream.__aiter__()
        source_done = False
        while True:
            try:
                while True:
                    yield injected.get_nowait()
            except asyncio.QueueEmpty:
                pass

            if source_done:
                try:
                    yield await asyncio.wait_for(injected.get(), timeout=0.2)
                    continue
                except asyncio.TimeoutError:
                    return

            try:
                yield await source.__anext__()
            except StopAsyncIteration:
                source_done = True

    async def converse_stream(
        self,
        input_stream: AsyncIterator[VoiceLLMInputEvent],
        session_config: VoiceLLMSessionConfig | None = None,
    ) -> AsyncIterator[VoiceLLMOutputEvent]:
        if self.model_plugin is None:
            raise RuntimeError("persona model plugin is not initialized")
        session_config = session_config or VoiceLLMSessionConfig()
        model_session_config = replace(
            session_config,
            system_prompt=self._persona_system_prompt(session_config),
            tools=PERSONA_TOOL_DEFINITIONS,
        )
        injected: asyncio.Queue[VoiceLLMInputEvent] = asyncio.Queue()
        pending_partials: list[str] = []
        turn_transcripts: list[str] = []
        pending_task_starts: list[PendingSubAgentTask] = []
        background_tasks: set[asyncio.Task[None]] = set()
        task_events: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        remove_task_event_listener = None

        def enqueue_task_event(task: dict[str, Any], event: dict[str, Any]) -> None:
            if str(task.get("session_id") or "") != str(session_config.session_id or ""):
                return
            task_events.put_nowait(self._task_event_payload(task, event))

        if hasattr(self.task_runtime, "add_event_listener"):
            remove_task_event_listener = self.task_runtime.add_event_listener(enqueue_task_event)  # type: ignore[union-attr]

        def schedule_task_start(pending: PendingSubAgentTask) -> None:
            task = asyncio.create_task(self._run_async_task(pending, injected))
            background_tasks.add(task)
            task.add_done_callback(background_tasks.discard)

        model_event_task: asyncio.Task[VoiceLLMOutputEvent] | None = None
        task_event_task: asyncio.Task[dict[str, Any]] | None = None
        try:
            model_events = self.model_plugin.converse_stream(
                self._merged_input_stream(input_stream, injected),
                session_config=model_session_config,
            ).__aiter__()
            model_event_task = asyncio.create_task(model_events.__anext__())
            task_event_task = asyncio.create_task(task_events.get())

            while model_event_task is not None:
                wait_set = {model_event_task}
                if task_event_task is not None:
                    wait_set.add(task_event_task)
                done, _pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

                if task_event_task is not None and task_event_task in done:
                    yield VoiceLLMOutputEvent(task_event=task_event_task.result())
                    for payload in self._drain_task_events(task_events):
                        yield VoiceLLMOutputEvent(task_event=payload)
                    task_event_task = asyncio.create_task(task_events.get())
                    if model_event_task not in done:
                        continue

                if model_event_task not in done:
                    continue

                try:
                    event = model_event_task.result()
                except StopAsyncIteration:
                    model_event_task = None
                    break

                self._log_model_event(session_config.session_id, event)
                if event.user_transcript:
                    turn_transcripts.append(event.user_transcript)
                    yield VoiceLLMOutputEvent(
                        user_transcript=event.user_transcript,
                        question_id=event.question_id,
                        reply_id=event.reply_id,
                    )
                    event = replace(event, user_transcript="")
                    if not event.tool_calls and not event.barge_in and not self._has_assistant_output(event):
                        model_event_task = asyncio.create_task(model_events.__anext__())
                        continue

                if event.tool_calls:
                    for call in event.tool_calls:
                        name = call.name.strip()
                        if name == "wait_for_more_input":
                            partial_text = self._partial_text_for_wait(call, turn_transcripts)
                            if partial_text:
                                pending_partials.append(partial_text)
                            turn_transcripts.clear()
                            try:
                                tool_result = await self._execute_tool(call, session_config.session_id)
                                result = tool_result.result
                            except Exception as exc:
                                logger.exception("persona wait tool call failed: %s", call.name)
                                result = {"ok": False, "error": str(exc)}
                            await injected.put(
                                VoiceLLMInputEvent(
                                    tool_result=ToolResult(
                                        id=call.id,
                                        name=call.name,
                                        result=result,
                                        suppress_response=True,
                                    )
                                )
                            )
                            continue

                        final_user_text = self._final_user_text(call, pending_partials, turn_transcripts)
                        effective_call = call
                        if name == "create_task" and final_user_text:
                            args = dict(call.arguments or {})
                            args["description"] = final_user_text
                            effective_call = ToolCall(id=call.id, name=call.name, arguments=args)
                        pending_partials.clear()
                        turn_transcripts.clear()

                        try:
                            tool_result = await self._execute_tool(effective_call, session_config.session_id)
                            if tool_result.pending_task is not None:
                                pending_task_starts.append(tool_result.pending_task)
                            result = tool_result.result
                        except Exception as exc:
                            logger.exception("persona tool call failed: %s", call.name)
                            result = {"ok": False, "error": str(exc)}
                        await injected.put(
                            VoiceLLMInputEvent(
                                tool_result=ToolResult(
                                    id=call.id,
                                    name=call.name,
                                    result=result,
                                )
                            )
                        )
                    for payload in self._drain_task_events(task_events):
                        yield VoiceLLMOutputEvent(task_event=payload)
                    model_event_task = asyncio.create_task(model_events.__anext__())
                    continue

                if self._has_assistant_output(event) and (pending_partials or turn_transcripts):
                    pending_partials.clear()
                    turn_transcripts.clear()
                yield event

                if event.is_final and pending_task_starts:
                    starts = pending_task_starts[:]
                    pending_task_starts.clear()
                    for pending in starts:
                        schedule_task_start(pending)
                model_event_task = asyncio.create_task(model_events.__anext__())
            for payload in self._drain_task_events(task_events):
                yield VoiceLLMOutputEvent(task_event=payload)
        finally:
            for task in (model_event_task, task_event_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (model_event_task, task_event_task) if task is not None),
                return_exceptions=True,
            )
            if remove_task_event_listener is not None:
                remove_task_event_listener()
            for task in background_tasks:
                task.cancel()
            if background_tasks:
                await asyncio.gather(*background_tasks, return_exceptions=True)
