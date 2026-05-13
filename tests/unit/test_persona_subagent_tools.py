from datetime import datetime, timezone

import pytest

from inference.plugins.voice_llm.persona.schemas import Task
from inference.plugins.voice_llm.persona.subagents.agent import _default_system_prompt
from inference.plugins.voice_llm.persona.subagents.default_tools import build_default_subagent_tools
from inference.plugins.voice_llm.persona.i18n import Localizer
from inference.plugins.voice_llm.persona.tools import ZhihuClient, ZhihuConfig, ZhihuToolExecutor


class CallbackRecorder:
    def __init__(self):
        self.artifacts = []
        self.events = []

    async def event(self, task_id, event):
        self.events.append((task_id, event))

    async def artifact(self, task_id, artifact):
        self.artifacts.append((task_id, artifact))
        return {"id": "artifact-1"}


class FakeModel:
    provider = "fake"
    model = "fake-model"


class FakeExecutor:
    async def execute(self, name, arguments):
        return {"ok": False, "tool": name, "error": "no sources"}


def _task() -> Task:
    return Task(
        id="task-1",
        session_id="session-1",
        title="知乎热点",
        user_request="今天知乎有哪些热门信息",
        created_at=datetime.now(timezone.utc),
    )


def test_subagent_prompt_requires_terminal_report_on_no_sources():
    prompt = _default_system_prompt(_task(), Localizer("zh-CN"), [])

    assert "工具错误" in prompt
    assert "最终产物工具" in prompt
    assert "不要无限重试" in prompt
    assert "caveats" in prompt


@pytest.mark.asyncio
async def test_create_html_report_supports_no_source_artifact():
    callbacks = CallbackRecorder()
    context = {"tool_trace": [{"tool": "hot_list", "arguments": {"limit": 10}, "iteration": 1}]}
    tools = build_default_subagent_tools(
        task=_task(),
        tool_executor=FakeExecutor(),
        callbacks=callbacks,
        model=FakeModel(),
        tool_runtime_context=context,
    )
    report_tool = next(tool for tool in tools if tool.name == "create_html_report")

    result = await report_tool.ainvoke(
        {
            "title": "知乎热点",
            "summary": "没有找到可确认的实时来源。",
            "sections": [
                {
                    "heading": "结果",
                    "paragraphs": ["当前工具没有返回可用来源。"],
                }
            ],
            "sources": [],
            "caveats": ["本报告没有可打开来源，不能视为实时事实。"],
        }
    )

    assert result == {
        "ok": True,
        "artifact_id": "artifact-1",
        "summary": "没有找到可确认的实时来源。",
    }
    assert len(callbacks.artifacts) == 1
    _, artifact = callbacks.artifacts[0]
    assert artifact.type == "html"
    assert artifact.mime_type == "text/html; charset=utf-8"
    assert artifact.metadata["source_count"] == 0
    assert artifact.metadata["tool_trace"] == context["tool_trace"]
    assert "未提供可打开来源" in artifact.content
    assert "本报告没有可打开来源" in artifact.content
    assert callbacks.events[0][1].event_type == "task.completed"
    assert callbacks.events[0][1].payload == {"artifact_id": "artifact-1"}


@pytest.mark.asyncio
async def test_zhihu_tool_executor_returns_clear_missing_secret_error():
    executor = ZhihuToolExecutor(ZhihuClient(ZhihuConfig(access_secret="")))

    result = await executor.execute("hot_list", {"limit": 10})

    assert result["ok"] is False
    assert result["tool"] == "hot_list"
    assert "ZHIHU_ACCESS_SECRET" in result["error"]
