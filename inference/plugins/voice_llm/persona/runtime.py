from __future__ import annotations

import asyncio
import inspect
import logging
import uuid
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from inference.plugins.voice_llm.persona.i18n import Localizer
from inference.plugins.voice_llm.persona.llm import AgentLLM, build_agent_llm_from_runtime_config
from inference.plugins.voice_llm.persona.schemas import Artifact, ArtifactRequest, Task, TaskEvent
from inference.plugins.voice_llm.persona.subagents.default_tools import run_task_with_langgraph
from inference.plugins.voice_llm.persona.tools import (
    NullSearchTool,
    SearchTool,
    ZhihuClient,
    ZhihuToolExecutor,
    zhihu_config_from_runtime_config,
)

logger = logging.getLogger(__name__)


ACTIVE_STATUSES = {"queued", "running", "waiting_user"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_json(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json", exclude_none=True)
    return dict(model)


def _default_title(user_request: str) -> str:
    title = user_request.strip() or "后台任务"
    if len(title) > 48:
        return title[:48]
    return title


def _persona_runtime_params(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    inference = runtime_config.get("inference", {}) if isinstance(runtime_config, dict) else {}
    inference = inference if isinstance(inference, dict) else {}
    persona_agent = inference.get("persona_agent", {})
    if isinstance(persona_agent, dict) and persona_agent:
        return persona_agent
    persona_section = inference.get("persona", {})
    persona_section = persona_section if isinstance(persona_section, dict) else {}
    persona_plugin = persona_section.get("persona", {})
    return persona_plugin if isinstance(persona_plugin, dict) else {}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, parsed)


class RuntimeCallbacks:
    def __init__(self, runtime: LocalTaskRuntime) -> None:
        self.runtime = runtime

    async def event(self, task_id: str, event: TaskEvent) -> None:
        await self.runtime.append_event(task_id, event)

    async def artifact(self, task_id: str, artifact: ArtifactRequest) -> dict[str, Any]:
        return await self.runtime.create_artifact(task_id, artifact)


class LocalTaskRuntime:
    """PersonaAgent-owned task runtime.

    This replaces the previous HTTP Agent Worker boundary. PersonaAgent creates
    task records in memory, runs the matching sub-agent graph in this process,
    and keeps the event/artifact context available to the supervisor graph.
    """

    def __init__(
        self,
        *,
        runtime_config: dict[str, Any] | None = None,
        llm: AgentLLM | None = None,
        search_tool: SearchTool | None = None,
        tool_executor: ZhihuToolExecutor | None = None,
        max_active_tasks_per_session: int = 3,
    ) -> None:
        persona_params = _persona_runtime_params(runtime_config)
        self.llm = llm or build_agent_llm_from_runtime_config(runtime_config)
        self.search_tool = search_tool or NullSearchTool()
        self.tool_executor = tool_executor or ZhihuToolExecutor(
            ZhihuClient(zhihu_config_from_runtime_config(runtime_config))
        )
        self.max_agent_iterations = _positive_int(persona_params.get("max_agent_iterations"), 8)
        self.max_active_tasks_per_session = max(1, max_active_tasks_per_session)
        self._tasks: dict[str, Task] = {}
        self._events: dict[str, list[TaskEvent]] = {}
        self._artifacts: dict[str, Artifact] = {}
        self._task_artifacts: dict[str, list[str]] = {}
        self._runners: dict[str, asyncio.Task[None]] = {}
        self._event_listeners: set[Callable[[dict[str, Any], dict[str, Any]], Any]] = set()
        self._lock = asyncio.Lock()

    async def shutdown(self) -> None:
        runners = [task for task in self._runners.values() if not task.done()]
        for task in runners:
            task.cancel()
        if runners:
            await asyncio.gather(*runners, return_exceptions=True)
        self._runners.clear()

    def add_event_listener(self, listener: Callable[[dict[str, Any], dict[str, Any]], Any]) -> Callable[[], None]:
        self._event_listeners.add(listener)

        def remove() -> None:
            self._event_listeners.discard(listener)

        return remove

    async def _notify_event_listeners(self, task: dict[str, Any], event: dict[str, Any]) -> None:
        if not self._event_listeners:
            return
        for listener in list(self._event_listeners):
            try:
                result = listener(task, event)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception("persona task event listener failed: task_id=%s", task.get("id"))

    async def create_task(self, session_id: str, args: dict[str, Any]) -> dict[str, Any]:
        user_request = str(
            args.get("user_request")
            or args.get("description")
            or args.get("request")
            or ""
        ).strip()
        if not user_request:
            raise ValueError("create_task requires user_request")
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("create_task requires session_id")

        title = _default_title(user_request)
        metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else None
        locale = str(args.get("locale") or "").strip() or None
        now = _now()
        task = Task(
            id=str(uuid.uuid4()),
            session_id=session_id,
            character_id=str(args.get("character_id") or "").strip() or None,
            title=title,
            user_request=user_request,
            status="queued",
            progress=0,
            locale=locale,
            metadata=metadata,
            created_at=now,
            updated_at=now,
        )

        async with self._lock:
            active = [
                existing
                for existing in self._tasks.values()
                if existing.session_id == session_id and existing.status in ACTIVE_STATUSES
            ]
            if len(active) >= self.max_active_tasks_per_session:
                raise RuntimeError(f"session already has {len(active)} active tasks")
            self._tasks[task.id] = task
            self._events[task.id] = []
            self._task_artifacts[task.id] = []

        await self.append_event(
            task.id,
            TaskEvent(
                event_type="task.queued",
                status="queued",
                message="任务已加入队列。",
                progress=0,
            ),
        )
        return _as_json(task)

    async def start_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            if task.status in TERMINAL_STATUSES:
                return _as_json(task)
            runner = self._runners.get(task_id)
            if runner is not None and not runner.done():
                return _as_json(task)
            runner = asyncio.create_task(self._run_task(task_id))
            self._runners[task_id] = runner
            runner.add_done_callback(lambda done, task_id=task_id: self._runners.pop(task_id, None))
            return _as_json(task)

    async def get_task(self, task_id: str) -> dict[str, Any]:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            return _as_json(task)

    async def get_task_events(self, task_id: str, after_seq: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        async with self._lock:
            events = self._events.get(task_id, [])
            selected = [
                event
                for event in events
                if int(event.seq or 0) > int(after_seq or 0)
            ][: max(1, min(limit, 500))]
            return [_as_json(event) for event in selected]

    async def get_task_status(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            tasks = sorted(
                [
                    task
                    for task in self._tasks.values()
                    if task.session_id == session_id and task.status in ACTIVE_STATUSES
                ],
                key=lambda task: task.updated_at or task.created_at or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            if not tasks:
                return {"task": None, "events": []}
            task = tasks[0]
            events = self._events.get(task.id, [])[-20:]
            return {"task": _as_json(task), "events": [_as_json(event) for event in events]}

    async def cancel_task(self, session_id: str) -> dict[str, Any]:
        status = await self.get_task_status(session_id)
        task = status.get("task")
        if not task:
            return {"cancelled": False, "reason": "no_active_task"}
        task_id = str(task.get("id") or "")
        runner = self._runners.get(task_id)
        if runner and not runner.done():
            runner.cancel()
        event = await self.append_event(
            task_id,
            TaskEvent(
                event_type="task.cancelled",
                status="cancelled",
                message=Localizer(task.get("locale")).text("worker.cancelled"),
                progress=int(task.get("progress") or 0),
            ),
        )
        return {"cancelled": True, "task": await self.get_task(task_id), "event": event}

    async def create_artifact(self, task_id: str, artifact_request: ArtifactRequest) -> dict[str, Any]:
        artifact = Artifact(
            id=str(uuid.uuid4()),
            task_id=task_id,
            type=artifact_request.type,
            title=artifact_request.title,
            mime_type=artifact_request.mime_type,
            content=artifact_request.content,
            metadata=artifact_request.metadata,
            created_at=_now(),
        )
        async with self._lock:
            if task_id not in self._tasks:
                raise KeyError(f"task not found: {task_id}")
            self._artifacts[artifact.id] = artifact
            self._task_artifacts.setdefault(task_id, []).append(artifact.id)

        await self.append_event(
            task_id,
            TaskEvent(
                event_type="artifact.created",
                status="running",
                message="已生成一份资料：" + artifact.title,
                progress=90,
                payload={
                    "artifact_id": artifact.id,
                    "title": artifact.title,
                    "type": artifact.type,
                    "mime_type": artifact.mime_type,
                    "content": artifact.content,
                },
            ),
        )
        return _as_json(artifact)

    async def append_event(self, task_id: str, event: TaskEvent) -> dict[str, Any]:
        task_json: dict[str, Any] | None = None
        event_json: dict[str, Any] | None = None
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"task not found: {task_id}")
            if task.status in TERMINAL_STATUSES:
                return _as_json(self._events.get(task_id, [])[-1]) if self._events.get(task_id) else {}

            events = self._events.setdefault(task_id, [])
            now = _now()
            status = event.status or task.status
            progress = event.progress
            if progress == 0 and task.progress > 0 and status != "queued":
                progress = task.progress
            if status == "completed" and progress < 100:
                progress = 100

            stored = TaskEvent(
                task_id=task_id,
                seq=len(events) + 1,
                event_type=event.event_type.strip(),
                status=status,
                message=event.message.strip(),
                progress=progress,
                payload=event.payload,
                created_at=now,
            )
            events.append(stored)

            task.status = status  # type: ignore[assignment]
            task.progress = progress
            task.updated_at = now
            if status == "completed" and stored.message:
                task.result_summary = stored.message
            if status in TERMINAL_STATUSES:
                task.finished_at = now
            task_json = _as_json(task)
            event_json = _as_json(stored)
        await self._notify_event_listeners(task_json, event_json)
        return event_json

    async def _run_task(self, task_id: str) -> None:
        try:
            task = self._tasks[task_id]
            await self.append_event(
                task.id,
                TaskEvent(
                    event_type="task.started",
                    status="running",
                    message="后台任务已启动。",
                    progress=5,
                ),
            )
            task = self._tasks[task_id]
            await run_task_with_langgraph(
                task,
                self.search_tool,
                RuntimeCallbacks(self),
                llm=self.llm,
                tool_executor=self.tool_executor,
                max_agent_iterations=self.max_agent_iterations,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("persona local task failed: task_id=%s", task_id)
            task = self._tasks.get(task_id)
            localizer = Localizer(task.locale if task else None)
            try:
                await self.append_event(
                    task_id,
                    TaskEvent(
                        event_type="task.failed",
                        status="failed",
                        message=localizer.text("worker.failed", error=str(exc)),
                        progress=task.progress if task else 0,
                    ),
                )
            except Exception:
                logger.exception("failed to record persona local task failure: task_id=%s", task_id)
