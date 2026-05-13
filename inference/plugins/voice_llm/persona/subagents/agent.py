from __future__ import annotations

import json
import operator
from typing import Annotated, Any, Literal, Protocol, TypedDict

from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool
from langgraph.graph import END, START, StateGraph

from inference.plugins.voice_llm.persona.i18n import Localizer, locale_from_metadata, normalize_locale
from inference.plugins.voice_llm.persona.schemas import Task, TaskEvent


class TaskCallbacks(Protocol):
    async def event(self, task_id: str, event: TaskEvent) -> None:
        ...

    async def artifact(self, task_id: str, artifact: Any) -> dict[str, Any]:
        ...


class SubAgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    tool_trace: Annotated[list[dict[str, Any]], operator.add]
    completed: bool


def _localizer_for_task(task: Task) -> Localizer:
    if task.locale:
        return Localizer(normalize_locale(task.locale))
    return Localizer(locale_from_metadata(task.metadata))


def _safe_json(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _model_provider(model: Any) -> str:
    return str(
        getattr(model, "model_provider", None)
        or getattr(model, "provider", None)
        or model.__class__.__name__
    )


def _model_name(model: Any) -> str:
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or getattr(model, "model_id", None)
        or model.__class__.__name__
    )


def _initial_messages(task: Task, localizer: Localizer) -> list[AnyMessage]:
    return [
        HumanMessage(
            content=json.dumps(
                {
                    "task_id": task.id,
                    "title": task.title,
                    "user_request": task.user_request,
                    "locale": localizer.locale,
                },
                ensure_ascii=False,
            ),
        )
    ]


def _default_system_prompt(task: Task, localizer: Localizer, tools: list[BaseTool]) -> str:
    tool_names = ", ".join(tool.name for tool in tools)
    return "\n".join(
        [
            "你是 CyberVerse 的通用后台 SubAgent。",
            "你必须根据用户任务和可用工具自行决定执行过程，不要按固定工作流机械执行。",
            f"可用工具包括：{tool_names}。",
            "需要获取外部信息时调用合适的查询工具；需要生成最终交付物时调用报告或产物生成工具。",
            "如果查询工具返回空结果或工具错误，也必须调用最终产物工具，清楚说明未找到可靠来源、失败原因和后续建议，不要无限重试。",
            "不要把无来源内容伪装成事实；来源不足时在 caveats 中说明。",
            "如果用户请求是中文，最终内容应优先使用简洁中文。",
            f"任务语言环境：{localizer.locale}。",
            f"任务标题：{task.title}。",
        ]
    )


def _tool_call_name(tool_call: dict[str, Any]) -> str:
    return str(tool_call.get("name") or "").strip()


def _tool_call_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    args = tool_call.get("args")
    return dict(args) if isinstance(args, dict) else {}


def _tool_call_id(tool_call: dict[str, Any], index: int) -> str:
    return str(tool_call.get("id") or f"tool-call-{index}")


async def run_subagent(
    *,
    task: Task,
    model: Any,
    tools: list[BaseTool],
    callbacks: TaskCallbacks,
    max_agent_iterations: int = 8,
    terminal_tool_names: set[str] | None = None,
    tool_labels: dict[str, str] | None = None,
    tool_runtime_context: dict[str, Any] | None = None,
    system_prompt: str | None = None,
) -> None:
    if not callable(getattr(model, "bind_tools", None)):
        raise RuntimeError("agent model does not support bind_tools")
    localizer = _localizer_for_task(task)
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)
    terminal_tool_names = terminal_tool_names or set()
    tool_labels = tool_labels or {}
    tool_runtime_context = tool_runtime_context if tool_runtime_context is not None else {}
    max_agent_iterations = max(1, max_agent_iterations)

    await callbacks.event(
        task.id,
        TaskEvent(
            event_type="plan.created",
            status="running",
            message="SubAgent 将自主选择工具并执行后台任务。",
            progress=15,
            payload={
                "mode": "tool_calling_agent",
                "tools": [tool.name for tool in tools],
                "locale": localizer.locale,
                "llm_provider": _model_provider(model),
                "llm_model": _model_name(model),
            },
        ),
    )

    prompt = system_prompt or _default_system_prompt(task, localizer, tools)

    async def llm_call(state: SubAgentState) -> SubAgentState:
        message = await model_with_tools.ainvoke([SystemMessage(content=prompt)] + state.get("messages", []))
        return {
            "messages": [message],
            "llm_calls": int(state.get("llm_calls") or 0) + 1,
        }

    async def tool_node(state: SubAgentState) -> SubAgentState:
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        calls = getattr(last_message, "tool_calls", []) or []
        tool_messages: list[AnyMessage] = []
        new_trace: list[dict[str, Any]] = []
        completed = False
        iteration = int(state.get("llm_calls") or 0)
        existing_trace = list(state.get("tool_trace") or [])
        for index, call in enumerate(calls):
            name = _tool_call_name(call)
            arguments = _tool_call_args(call)
            call_id = _tool_call_id(call, index)
            safe_arguments = {key: value for key, value in arguments.items() if key != "_raw"}
            await callbacks.event(
                task.id,
                TaskEvent(
                    event_type="agent.tool_call",
                    status="running",
                    message=f"调用{tool_labels.get(name, name)}。",
                    progress=min(85, 15 + iteration * 8),
                    payload={"tool": name, "arguments": safe_arguments, "iteration": iteration},
                ),
            )
            trace_entry = {"tool": name, "arguments": safe_arguments, "iteration": iteration}
            new_trace.append(trace_entry)
            tool_runtime_context["tool_trace"] = [*existing_trace, *new_trace]

            tool = tools_by_name.get(name)
            if tool is None:
                result = {"ok": False, "tool": name, "error": f"unsupported tool: {name}"}
            else:
                result = await tool.ainvoke(arguments)

            tool_messages.append(ToolMessage(content=_safe_json(result), name=name, tool_call_id=call_id))
            if name in terminal_tool_names:
                completed = True
                break

        return {
            "messages": tool_messages,
            "tool_trace": new_trace,
            "completed": completed,
        }

    def route_after_llm(state: SubAgentState) -> Literal["tool_node", "__end__"]:
        if state.get("completed"):
            return END
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        if getattr(last_message, "tool_calls", None):
            return "tool_node"
        return END

    def route_after_tools(state: SubAgentState) -> Literal["llm_call", "__end__"]:
        if state.get("completed"):
            return END
        if int(state.get("llm_calls") or 0) >= max_agent_iterations:
            return END
        return "llm_call"

    agent_builder = StateGraph(SubAgentState)
    agent_builder.add_node("llm_call", llm_call)
    agent_builder.add_node("tool_node", tool_node)
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges("llm_call", route_after_llm, ["tool_node", END])
    agent_builder.add_conditional_edges("tool_node", route_after_tools, ["llm_call", END])
    agent = agent_builder.compile()

    final_state = await agent.ainvoke(
        {
            "messages": _initial_messages(task, localizer),
            "llm_calls": 0,
            "tool_trace": [],
            "completed": False,
        }
    )
    if terminal_tool_names and not final_state.get("completed"):
        raise TimeoutError(f"subagent did not call a terminal tool within {max_agent_iterations} LLM calls")
