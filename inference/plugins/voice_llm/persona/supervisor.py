from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, TypedDict

from inference.core.types import ToolCall
from inference.plugins.voice_llm.persona.runtime import LocalTaskRuntime, TERMINAL_STATUSES

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict, total=False):
    call: ToolCall
    session_id: str
    route: str
    normalized_args: dict[str, Any]
    result: dict[str, Any]
    pending_task: dict[str, Any]


@dataclass
class PendingSubAgentTask:
    session_id: str
    args: dict[str, Any]
    user_request: str
    task_id: str = ""


@dataclass
class SupervisorToolResult:
    result: dict[str, Any]
    pending_task: PendingSubAgentTask | None = None


def _task_description_from_args(args: dict[str, Any]) -> str:
    return str(
        args.get("description")
        or args.get("user_request")
        or args.get("request")
        or args.get("text")
        or ""
    ).strip()


def _normalize_create_task_args(args: dict[str, Any]) -> dict[str, Any]:
    description = _task_description_from_args(args)
    normalized = dict(args)
    normalized["description"] = description
    normalized["user_request"] = description
    normalized.pop("kind", None)
    normalized.pop("title", None)
    return normalized


class PersonaSupervisor:
    """Top-level PersonaAgent supervisor graph.

    The graph owns tool-call routing and local task orchestration decisions.
    Long-running sub-agent execution is returned as a pending task so the voice
    layer can let the realtime model speak an ACK before the background work
    begins.
    """

    def __init__(
        self,
        *,
        runtime: LocalTaskRuntime,
        checkpoint_db_path: str = "",
        task_poll_interval_seconds: float = 1.0,
        task_monitor_timeout_seconds: float = 1800.0,
    ) -> None:
        self.runtime = runtime
        self.checkpoint_db_path = checkpoint_db_path
        self.task_poll_interval_seconds = max(0.1, task_poll_interval_seconds)
        self.task_monitor_timeout_seconds = max(1.0, task_monitor_timeout_seconds)
        self._graph: Any | None = None
        self._checkpoint_conn: Any | None = None

    async def initialize(self) -> None:
        self._graph = await self._compile_graph()

    async def shutdown(self) -> None:
        if self._checkpoint_conn is not None:
            await self._checkpoint_conn.close()
            self._checkpoint_conn = None
        await self.runtime.shutdown()

    async def _compile_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception:
            return None

        graph = StateGraph(SupervisorState)

        async def normalize_tool_call(state: SupervisorState) -> SupervisorState:
            call = state["call"]
            name = call.name.strip()
            args = dict(call.arguments or {})
            if name == "create_task":
                args = _normalize_create_task_args(args)
            return {"route": name, "normalized_args": args}

        async def execute_route(state: SupervisorState) -> SupervisorState:
            route = state["route"]
            args = state.get("normalized_args") or {}
            session_id = state["session_id"]
            if route == "wait_for_more_input":
                return {
                    "result": {
                        "ok": True,
                        "waiting": True,
                        "partial_text": str(args.get("partial_text") or "").strip(),
                    }
                }
            if route == "create_task":
                user_request = str(args.get("user_request") or "").strip()
                if not user_request:
                    raise ValueError("create_task requires description")
                task = await self.runtime.create_task(session_id, args)
                return {
                    "result": self._accepted_task_result(task),
                    "pending_task": {
                        "session_id": session_id,
                        "args": args,
                        "user_request": user_request,
                        "task_id": str(task.get("id") or ""),
                    },
                }
            if route == "get_task_status":
                return {"result": await self.runtime.get_task_status(session_id)}
            if route == "cancel_task":
                return {"result": await self.runtime.cancel_task(session_id)}
            raise ValueError(f"unsupported persona tool: {route}")

        graph.add_node("normalize_tool_call", normalize_tool_call)
        graph.add_node("execute_route", execute_route)
        graph.add_edge(START, "normalize_tool_call")
        graph.add_edge("normalize_tool_call", "execute_route")
        graph.add_edge("execute_route", END)

        checkpointer = None
        if self.checkpoint_db_path:
            try:
                import aiosqlite
                from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                if self.checkpoint_db_path != ":memory:":
                    os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_db_path)), exist_ok=True)
                self._checkpoint_conn = await aiosqlite.connect(self.checkpoint_db_path)
                checkpointer = AsyncSqliteSaver(self._checkpoint_conn)
                await checkpointer.setup()
            except Exception as exc:
                logger.warning("persona supervisor checkpoint disabled: %s", exc)
                if self._checkpoint_conn is not None:
                    await self._checkpoint_conn.close()
                    self._checkpoint_conn = None
        if checkpointer is None:
            return graph.compile()
        return graph.compile(checkpointer=checkpointer)

    async def handle_tool_call(self, call: ToolCall, session_id: str) -> SupervisorToolResult:
        if not session_id and call.name.strip() != "wait_for_more_input":
            raise ValueError("persona tool execution requires session_id")

        if self._graph is None:
            state = await self._execute_without_graph(call, session_id)
        else:
            state = await self._graph.ainvoke(
                {"call": call, "session_id": session_id},
                config={"configurable": {"thread_id": f"{session_id}:{call.id or call.name}"}},
            )
        pending = state.get("pending_task") or None
        pending_task = None
        if isinstance(pending, dict):
            pending_task = PendingSubAgentTask(
                session_id=str(pending.get("session_id") or ""),
                args=dict(pending.get("args") or {}),
                user_request=str(pending.get("user_request") or "").strip(),
                task_id=str(pending.get("task_id") or "").strip(),
            )
        return SupervisorToolResult(result=dict(state.get("result") or {}), pending_task=pending_task)

    async def _execute_without_graph(self, call: ToolCall, session_id: str) -> SupervisorState:
        name = call.name.strip()
        args = dict(call.arguments or {})
        if name == "wait_for_more_input":
            return {
                "result": {
                    "ok": True,
                    "waiting": True,
                    "partial_text": str(args.get("partial_text") or "").strip(),
                }
            }
        if name == "create_task":
            args = _normalize_create_task_args(args)
            user_request = str(args.get("user_request") or "").strip()
            if not user_request:
                raise ValueError("create_task requires description")
            task = await self.runtime.create_task(session_id, args)
            return {
                "result": self._accepted_task_result(task),
                "pending_task": {
                    "session_id": session_id,
                    "args": args,
                    "user_request": user_request,
                    "task_id": str(task.get("id") or ""),
                },
            }
        if name == "get_task_status":
            return {"result": await self.runtime.get_task_status(session_id)}
        if name == "cancel_task":
            return {"result": await self.runtime.cancel_task(session_id)}
        raise ValueError(f"unsupported persona tool: {name}")

    async def run_pending_task(self, pending: PendingSubAgentTask) -> str:
        try:
            task_id = pending.task_id.strip()
            if not task_id:
                task = await self.runtime.create_task(pending.session_id, pending.args)
                task_id = str(task.get("id") or "").strip()
            if not task_id:
                raise RuntimeError("task runtime did not return a task id")
            if hasattr(self.runtime, "start_task"):
                await self.runtime.start_task(task_id)  # type: ignore[attr-defined]
            final_task, events = await self.wait_for_task_terminal(task_id)
            return self.task_completion_prompt(pending.user_request, final_task, events)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("persona supervisor task failed")
            return self.task_start_failed_prompt(pending.user_request, exc)

    async def wait_for_task_terminal(self, task_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        deadline = asyncio.get_running_loop().time() + self.task_monitor_timeout_seconds
        after_seq = 0
        events: list[dict[str, Any]] = []
        task = await self.runtime.get_task(task_id)
        while True:
            new_events = await self.runtime.get_task_events(task_id, after_seq=after_seq, limit=100)
            for event in new_events:
                events.append(event)
                try:
                    after_seq = max(after_seq, int(event.get("seq") or 0))
                except (TypeError, ValueError):
                    pass
            task = await self.runtime.get_task(task_id)
            if str(task.get("status") or "") in TERMINAL_STATUSES:
                return task, events
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"task {task_id} did not finish before persona monitor timeout")
            await asyncio.sleep(self.task_poll_interval_seconds)

    @staticmethod
    def _accepted_task_result(task: dict[str, Any] | None = None) -> dict[str, Any]:
        task_id = str((task or {}).get("id") or "").strip()
        return {
            "ok": True,
            "accepted": True,
            "status": "accepted",
            "reply": "好的，请稍等，我现在开始处理。",
            "task_id": task_id,
        }

    @staticmethod
    def latest_event_message(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            message = str(event.get("message") or "").strip()
            if message:
                return message
        return ""

    @staticmethod
    def latest_artifact_id(events: list[dict[str, Any]]) -> str:
        for event in reversed(events):
            payload = event.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = {}
            if isinstance(payload, dict):
                artifact_id = str(payload.get("artifact_id") or "").strip()
                if artifact_id:
                    return artifact_id
        return ""

    def task_completion_prompt(
        self,
        user_request: str,
        task: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> str:
        status = str(task.get("status") or "").strip()
        summary = str(task.get("result_summary") or "").strip() or self.latest_event_message(events)
        artifact_id = self.latest_artifact_id(events)
        artifact_hint = "资料已经在聊天侧生成，用户可以打开链接查看。" if artifact_id else "没有生成可打开的资料链接。"
        return "\n".join(
            [
                "后台任务结果已经返回。请作为数字人用自然口语回复用户，保持一到两句话。",
                f"用户原始请求：{user_request}",
                f"任务状态：{status}",
                f"结果摘要：{summary or '无'}",
                artifact_hint,
                "不要朗读内部字段名、JSON、任务 ID 或 artifact ID。",
            ]
        )

    @staticmethod
    def task_start_failed_prompt(user_request: str, error: Exception) -> str:
        return "\n".join(
            [
                "后台任务没有成功启动。请作为数字人用一句自然口语告诉用户稍后再试。",
                f"用户原始请求：{user_request}",
                f"错误原因：{error}",
            ]
        )
