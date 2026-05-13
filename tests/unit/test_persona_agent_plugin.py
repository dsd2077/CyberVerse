import asyncio
import logging

import pytest

from inference.core.types import (
    AudioChunk,
    PluginConfig,
    ToolCall,
    VoiceLLMInputEvent,
    VoiceLLMOutputEvent,
    VoiceLLMSessionConfig,
)
from inference.plugins.voice_llm.base import VoiceLLMPlugin
from inference.plugins.voice_llm.persona import runtime as runtime_module
from inference.plugins.voice_llm.persona.runtime import LocalTaskRuntime
from inference.plugins.voice_llm.persona.schemas import ArtifactRequest, TaskEvent
from inference.plugins.voice_llm.persona_agent import PERSONA_AGENT_INSTRUCTIONS, PersonaAgentPlugin
from langchain.messages import AIMessage


class FakeOmniPlugin(VoiceLLMPlugin):
    name = "omni.fake"

    async def initialize(self, config):
        self.scenario = config.params.get("scenario", "chat")

    async def shutdown(self):
        pass

    async def check_voice(self, session_config=None):
        pass

    async def _next_tool_result(self, input_stream):
        async for event in input_stream:
            if event.tool_result:
                return event.tool_result
        raise AssertionError("expected tool result")

    async def _next_text(self, input_stream):
        async for event in input_stream:
            if event.text:
                return event.text
        raise AssertionError("expected injected text")

    async def _next_input(self, input_stream):
        async for event in input_stream:
            return event
        raise AssertionError("expected input event")

    async def converse_stream(self, input_stream, session_config=None):
        self.last_session_config = session_config
        async for _ in input_stream:
            break

        if self.scenario == "chat":
            yield VoiceLLMOutputEvent(user_transcript="你好")
            yield VoiceLLMOutputEvent(
                transcript="你好，我在。",
                audio=AudioChunk(data=b"audio", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return

        if self.scenario == "memory_chat":
            assert "recall_memory" in [tool.name for tool in session_config.tools]
            yield VoiceLLMOutputEvent(user_transcript="记住我喜欢用 Pixi 管理环境")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="memory-1",
                        name="recall_memory",
                        arguments={"query": "记住我喜欢用 Pixi 管理环境"},
                    )
                ]
            )
            memory_result = await self._next_tool_result(input_stream)
            assert memory_result.name == "recall_memory"
            assert "Pixi" in memory_result.result["memories_text"]
            yield VoiceLLMOutputEvent(
                transcript="我记住了，你喜欢用 Pixi 管理环境。",
                audio=AudioChunk(data=b"memory", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return

        if self.scenario == "wait_for_more_input":
            yield VoiceLLMOutputEvent(user_transcript="我觉得...")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="wait-1",
                        name="wait_for_more_input",
                        arguments={"partial_text": "我觉得...", "reason": "unfinished thought"},
                    )
                ]
            )
            async for event in input_stream:
                if event.tool_result:
                    assert event.tool_result.name == "wait_for_more_input"
                    assert event.tool_result.suppress_response is True
                    return

        if self.scenario == "partial_create_task":
            yield VoiceLLMOutputEvent(user_transcript="我想让你帮我查一下")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="wait-1",
                        name="wait_for_more_input",
                        arguments={"partial_text": "我想让你帮我查一下", "reason": "missing object"},
                    )
                ]
            )
            wait_result = await self._next_tool_result(input_stream)
            assert wait_result.name == "wait_for_more_input"
            assert wait_result.suppress_response is True
            await self._next_input(input_stream)
            yield VoiceLLMOutputEvent(user_transcript="今天知乎有哪些热门信息")
            yield VoiceLLMOutputEvent(
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="create_task",
                        arguments={"description": "今天知乎有哪些热门信息"},
                    )
                ]
            )
            tool_result = await self._next_tool_result(input_stream)
            assert tool_result.name == "create_task"
            assert tool_result.result["accepted"] is True
            yield VoiceLLMOutputEvent(
                transcript="好的，请稍等。",
                audio=AudioChunk(data=b"ack", sample_rate=16000, is_final=True),
                is_final=True,
            )
            final_prompt = await self._next_text(input_stream)
            assert "用户原始请求：我想让你帮我查一下今天知乎有哪些热门信息" in final_prompt
            assert "任务状态：completed" in final_prompt
            yield VoiceLLMOutputEvent(
                transcript="查好了，资料已经整理好。",
                audio=AudioChunk(data=b"done", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return

        tool_name = self.scenario
        transcript_by_tool = {
            "create_task": "今天知乎有哪些热门信息",
            "create_task_search_kind": "查一下今天知乎有什么热门新闻。",
            "legacy_create_task": "今天知乎有哪些热门信息",
            "get_task_status": "现在查得怎么样了",
            "cancel_task": "算了不用查了",
        }
        if tool_name in transcript_by_tool:
            yield VoiceLLMOutputEvent(user_transcript=transcript_by_tool[tool_name])
        emitted_tool_name = "create_task" if tool_name in {"create_task_search_kind", "legacy_create_task"} else tool_name
        if tool_name == "create_task":
            tool_arguments = {"description": "今天知乎有哪些热门信息"}
            expected_request = "今天知乎有哪些热门信息"
        elif tool_name == "create_task_search_kind":
            tool_arguments = {"description": "查询今天知乎上的热门新闻", "kind": "search"}
            expected_request = "查询今天知乎上的热门新闻"
        elif tool_name == "legacy_create_task":
            tool_arguments = {"user_request": "今天知乎有哪些热门信息", "title": "知乎热点", "kind": "search"}
            expected_request = "今天知乎有哪些热门信息"
        else:
            tool_arguments = {"user_request": "今天知乎有哪些热门信息", "title": "知乎热点"}
            expected_request = "今天知乎有哪些热门信息"
        yield VoiceLLMOutputEvent(
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name=emitted_tool_name,
                    arguments=tool_arguments,
                )
            ]
        )
        tool_result = await self._next_tool_result(input_stream)
        assert tool_result.name == emitted_tool_name
        if emitted_tool_name == "create_task":
            assert tool_result.result["accepted"] is True
            yield VoiceLLMOutputEvent(
                transcript="好的，请稍等。",
                audio=AudioChunk(data=b"ack", sample_rate=16000, is_final=True),
                is_final=True,
            )
            final_prompt = await self._next_text(input_stream)
            assert f"用户原始请求：{expected_request}" in final_prompt
            assert "任务状态：completed" in final_prompt
            yield VoiceLLMOutputEvent(
                transcript="查好了，资料已经整理好。",
                audio=AudioChunk(data=b"done", sample_rate=16000, is_final=True),
                is_final=True,
            )
            return
        yield VoiceLLMOutputEvent(
            transcript=f"{emitted_tool_name} ok",
            audio=AudioChunk(data=b"ok", sample_rate=16000, is_final=True),
            is_final=True,
        )
        return


class FakeTaskClient:
    def __init__(self):
        self.calls = []

    async def create_task(self, session_id, args):
        self.calls.append(("create_task", session_id, args))
        return {"id": "task-1", "status": "queued"}

    async def get_task(self, task_id):
        self.calls.append(("get_task", task_id, {}))
        return {
            "id": task_id,
            "status": "completed",
            "progress": 100,
            "result_summary": "我已经整理好资料。",
        }

    async def get_task_events(self, task_id, after_seq=0, limit=100):
        self.calls.append(("get_task_events", task_id, {"after_seq": after_seq, "limit": limit}))
        if after_seq > 0:
            return []
        return [
            {
                "task_id": task_id,
                "seq": 1,
                "event_type": "task.completed",
                "status": "completed",
                "message": "我已经整理好资料。",
                "progress": 100,
                "payload": {"artifact_id": "artifact-1"},
            }
        ]

    async def get_task_status(self, session_id):
        self.calls.append(("get_task_status", session_id, {}))
        return {"task": {"id": "task-1", "status": "running", "progress": 30}, "events": []}

    async def cancel_task(self, session_id):
        self.calls.append(("cancel_task", session_id, {}))
        return {"cancelled": True, "task": {"id": "task-1", "status": "cancelled"}}

    async def shutdown(self):
        pass


class FakeRuntimeLLM:
    provider = "fake"
    model = "fake-runtime-llm"

    def bind_tools(self, tools):
        self.bound_tools = [tool.name for tool in tools]
        return self

    async def ainvoke(self, messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "report-1",
                    "name": "create_html_report",
                    "args": {
                        "title": "知乎热点",
                        "summary": "已整理好资料。",
                        "sections": [{"heading": "摘要", "paragraphs": ["已完成整理。"]}],
                        "sources": [{"title": "测试来源"}],
                    },
                }
            ]
        )


class FakeRuntimeLLMNoTerminal(FakeRuntimeLLM):
    async def ainvoke(self, messages):
        return AIMessage(content="我没有调用报告工具。", tool_calls=[])


class FakeTextRuntimeLLM:
    provider = "fake"
    model = "fake-text-llm"

    def __init__(self):
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        return AIMessage(content="暗号是紫藤-7391。")


class FakeRuntimeToolExecutor:
    client = None

    async def execute(self, name, arguments):
        return {"ok": True, "tool": name}


class FakeMemoryClient:
    enabled = True

    def __init__(self):
        self.recalls = []
        self.retains = []

    async def recall(self, query):
        self.recalls.append(query)
        return [{"text": "用户喜欢用 Pixi 管理环境。"}]

    async def retain(self, content, context="conversation"):
        self.retains.append({"content": content, "context": context})
        return {"ok": True}


async def make_persona(scenario, checkpoint_db_path):
    plugin = PersonaAgentPlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="persona.persona",
            params={
                "model_provider": "fake",
                "checkpoint_db_path": str(checkpoint_db_path),
                "task_poll_interval_seconds": 0.1,
                "task_monitor_timeout_seconds": 2,
            },
            shared={
                "omni": {
                    "fake": {
                        "plugin_class": "tests.unit.test_persona_agent_plugin.FakeOmniPlugin",
                        "scenario": scenario,
                    }
                },
                "runtime_config": {
                    "inference": {
                        "llm": {
                            "default": "qwen",
                            "qwen": {"api_key": "fake-key"},
                        }
                    }
                },
            },
        )
    )
    fake_runtime = FakeTaskClient()
    plugin.task_runtime = fake_runtime
    plugin.supervisor.runtime = fake_runtime
    return plugin


async def make_persona_with_local_runtime(scenario, checkpoint_db_path, monkeypatch):
    monkeypatch.setattr(
        runtime_module,
        "build_agent_llm_from_runtime_config",
        lambda _runtime_config=None: FakeRuntimeLLM(),
    )

    async def fake_run_task_with_langgraph(task, _search_tool, callbacks, **_kwargs):
        artifact = await callbacks.artifact(
            task.id,
            ArtifactRequest(
                title="知乎热点",
                type="html",
                mime_type="text/html; charset=utf-8",
                content="<html><body>已完成整理。</body></html>",
            ),
        )
        await callbacks.event(
            task.id,
            TaskEvent(
                event_type="task.completed",
                status="completed",
                message="已整理好资料。",
                progress=100,
                payload={"artifact_id": artifact["id"]},
            ),
        )

    monkeypatch.setattr(runtime_module, "run_task_with_langgraph", fake_run_task_with_langgraph)

    plugin = PersonaAgentPlugin()
    await plugin.initialize(
        PluginConfig(
            plugin_name="persona.persona",
            params={
                "model_provider": "fake",
                "checkpoint_db_path": str(checkpoint_db_path),
                "task_poll_interval_seconds": 0.01,
                "task_monitor_timeout_seconds": 2,
            },
            shared={
                "omni": {
                    "fake": {
                        "plugin_class": "tests.unit.test_persona_agent_plugin.FakeOmniPlugin",
                        "scenario": scenario,
                    }
                },
                "runtime_config": {"inference": {"persona_agent": {"max_agent_iterations": 1}}},
            },
        )
    )
    return plugin


async def one_input():
    yield VoiceLLMInputEvent(audio=b"pcm")


async def two_inputs():
    yield VoiceLLMInputEvent(audio=b"pcm-1")
    yield VoiceLLMInputEvent(audio=b"pcm-2")


async def text_input(text="我之前让你记住的暗号是什么？"):
    yield VoiceLLMInputEvent(text=text)


@pytest.mark.asyncio
async def test_persona_agent_passthrough_chat(tmp_path):
    plugin = await make_persona("chat", tmp_path / "persona.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "你好"
    assert outputs[1].transcript == "你好，我在。"
    assert outputs[1].audio is not None
    assert plugin.model_plugin.last_session_config.tools[0].name == "wait_for_more_input"
    assert plugin.model_plugin.last_session_config.tools[1].name == "create_task"
    assert "recall_memory" not in [tool.name for tool in plugin.model_plugin.last_session_config.tools]
    assert "PersonaAgent" in plugin.model_plugin.last_session_config.system_prompt
    assert "JSON" not in PERSONA_AGENT_INSTRUCTIONS
    assert plugin.task_runtime.calls == []


@pytest.mark.asyncio
async def test_persona_agent_text_input_uses_text_llm_and_hindsight(tmp_path):
    plugin = await make_persona("chat", tmp_path / "text-memory.db")
    memory = FakeMemoryClient()
    text_llm = FakeTextRuntimeLLM()
    plugin.memory_client = memory
    plugin.task_runtime.llm = text_llm

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                text_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert len(outputs) == 1
    assert outputs[0].transcript == "暗号是紫藤-7391。"
    assert outputs[0].is_final is True
    assert memory.recalls == ["我之前让你记住的暗号是什么？"]
    assert memory.retains == [
        {
            "content": "用户: 我之前让你记住的暗号是什么？\n助手: 暗号是紫藤-7391。",
            "context": "conversation",
        }
    ]
    system_text = text_llm.messages[0].content
    assert "相关长期记忆" in system_text
    assert "用户喜欢用 Pixi 管理环境。" in system_text
    assert not hasattr(plugin.model_plugin, "last_session_config")


@pytest.mark.asyncio
async def test_persona_agent_recalls_and_retains_hindsight_memory(tmp_path):
    plugin = await make_persona("memory_chat", tmp_path / "memory.db")
    memory = FakeMemoryClient()
    plugin.memory_client = memory

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "记住我喜欢用 Pixi 管理环境"
    assert outputs[-1].transcript == "我记住了，你喜欢用 Pixi 管理环境。"
    assert memory.recalls == ["记住我喜欢用 Pixi 管理环境"]
    assert memory.retains == [
        {
            "content": "用户: 记住我喜欢用 Pixi 管理环境\n助手: 我记住了，你喜欢用 Pixi 管理环境。",
            "context": "conversation",
        }
    ]
    assert "记忆" in plugin.model_plugin.last_session_config.system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["get_task_status", "cancel_task"])
async def test_persona_agent_executes_hidden_tool_calls(tool_name, tmp_path):
    plugin = await make_persona(tool_name, tmp_path / f"{tool_name}.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[-1].transcript == f"{tool_name} ok"
    assert outputs[0].user_transcript
    assert plugin.task_runtime.calls[0][0] == tool_name
    assert plugin.task_runtime.calls[0][1] == "session-1"


@pytest.mark.asyncio
async def test_persona_agent_create_task_acks_then_runs_async_task(tmp_path):
    plugin = await make_persona("create_task", tmp_path / "create_task.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "今天知乎有哪些热门信息"
    assert outputs[1].transcript == "好的，请稍等。"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_runtime.calls[0] == (
        "create_task",
        "session-1",
        {
            "description": "今天知乎有哪些热门信息",
            "user_request": "今天知乎有哪些热门信息",
        },
    )
    assert any(call[0] == "get_task_events" for call in plugin.task_runtime.calls)


@pytest.mark.asyncio
async def test_persona_agent_ignores_legacy_task_kind(tmp_path):
    plugin = await make_persona("create_task_search_kind", tmp_path / "search_kind.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "查一下今天知乎有什么热门新闻。"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_runtime.calls[0] == (
        "create_task",
        "session-1",
        {
            "description": "查询今天知乎上的热门新闻",
            "user_request": "查询今天知乎上的热门新闻",
        },
    )


@pytest.mark.asyncio
async def test_persona_agent_accepts_legacy_create_task_args(tmp_path):
    plugin = await make_persona("legacy_create_task", tmp_path / "legacy.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_runtime.calls[0] == (
        "create_task",
        "session-1",
        {
            "description": "今天知乎有哪些热门信息",
            "user_request": "今天知乎有哪些热门信息",
        },
    )


@pytest.mark.asyncio
async def test_local_task_runtime_ignores_legacy_kind():
    runtime = LocalTaskRuntime(llm=FakeRuntimeLLM(), tool_executor=FakeRuntimeToolExecutor())

    try:
        task = await runtime.create_task(
            "session-1",
            {
                "description": "查询今天知乎上的热门新闻",
                "kind": "search",
            },
        )
    finally:
        await runtime.shutdown()

    assert "kind" not in task
    assert task["user_request"] == "查询今天知乎上的热门新闻"


@pytest.mark.asyncio
async def test_local_task_runtime_defers_background_start_until_explicit_start(monkeypatch):
    started = asyncio.Event()

    async def fake_run_task_with_langgraph(task, _search_tool, callbacks, **_kwargs):
        started.set()
        await callbacks.event(
            task.id,
            TaskEvent(
                event_type="task.completed",
                status="completed",
                message="已整理好资料。",
                progress=100,
            ),
        )

    monkeypatch.setattr(runtime_module, "run_task_with_langgraph", fake_run_task_with_langgraph)
    runtime = LocalTaskRuntime(llm=FakeRuntimeLLM(), tool_executor=FakeRuntimeToolExecutor())

    try:
        task = await runtime.create_task("session-1", {"description": "今天知乎有哪些热门信息"})
        await asyncio.sleep(0)
        events = await runtime.get_task_events(task["id"])

        assert [event["event_type"] for event in events] == ["task.queued"]
        assert not started.is_set()

        await runtime.start_task(task["id"])
        await asyncio.wait_for(started.wait(), timeout=1)
        events = await runtime.get_task_events(task["id"])
    finally:
        await runtime.shutdown()

    assert [event["event_type"] for event in events] == [
        "task.queued",
        "task.started",
        "task.completed",
    ]


@pytest.mark.asyncio
async def test_local_task_runtime_records_failed_terminal_event(monkeypatch):
    async def fake_run_task_with_langgraph(task, _search_tool, callbacks, **_kwargs):
        raise RuntimeError("missing search credentials")

    monkeypatch.setattr(runtime_module, "run_task_with_langgraph", fake_run_task_with_langgraph)
    runtime = LocalTaskRuntime(llm=FakeRuntimeLLM(), tool_executor=FakeRuntimeToolExecutor())

    try:
        task = await runtime.create_task("session-1", {"description": "今天知乎有哪些热门信息"})
        await runtime.start_task(task["id"])
        for _ in range(20):
            task = await runtime.get_task(task["id"])
            if task["status"] == "failed":
                break
            await asyncio.sleep(0.01)
        events = await runtime.get_task_events(task["id"])
    finally:
        await runtime.shutdown()

    assert task["status"] == "failed"
    assert events[-1]["event_type"] == "task.failed"
    assert "missing search credentials" in events[-1]["message"]


@pytest.mark.asyncio
async def test_local_task_runtime_cancels_active_task(monkeypatch):
    started = asyncio.Event()

    async def fake_run_task_with_langgraph(task, _search_tool, callbacks, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(runtime_module, "run_task_with_langgraph", fake_run_task_with_langgraph)
    runtime = LocalTaskRuntime(llm=FakeRuntimeLLM(), tool_executor=FakeRuntimeToolExecutor())

    try:
        task = await runtime.create_task("session-1", {"description": "今天知乎有哪些热门信息"})
        await runtime.start_task(task["id"])
        await asyncio.wait_for(started.wait(), timeout=1)
        result = await runtime.cancel_task("session-1")
        events = await runtime.get_task_events(task["id"])
    finally:
        await runtime.shutdown()

    assert result["cancelled"] is True
    assert result["task"]["status"] == "cancelled"
    assert events[-1]["event_type"] == "task.cancelled"


@pytest.mark.asyncio
async def test_local_task_runtime_marks_missing_terminal_report_as_failed():
    runtime = LocalTaskRuntime(llm=FakeRuntimeLLMNoTerminal(), tool_executor=FakeRuntimeToolExecutor())

    try:
        task = await runtime.create_task("session-1", {"description": "今天知乎有哪些热门信息"})
        await runtime.start_task(task["id"])
        for _ in range(20):
            task = await runtime.get_task(task["id"])
            if task["status"] == "failed":
                break
            await asyncio.sleep(0.01)
        events = await runtime.get_task_events(task["id"])
    finally:
        await runtime.shutdown()

    assert task["status"] == "failed"
    assert events[-1]["event_type"] == "task.failed"
    assert "terminal tool" in events[-1]["message"]


@pytest.mark.asyncio
async def test_persona_agent_projects_local_task_events(tmp_path, monkeypatch):
    plugin = await make_persona_with_local_runtime(
        "create_task",
        tmp_path / "local_events.db",
        monkeypatch,
    )

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    task_events = [event.task_event for event in outputs if event.task_event]
    event_types = [event["event_type"] for event in task_events]

    assert outputs[0].user_transcript == "今天知乎有哪些热门信息"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert "task.queued" in event_types
    assert "task.started" in event_types
    assert "artifact.created" in event_types
    assert "task.completed" in event_types
    assert all(event["type"] == "task_event" for event in task_events)
    assert all(event["session_id"] == "session-1" for event in task_events)
    assert any((event.get("payload") or {}).get("artifact_id") for event in task_events)
    ack_index = next(
        i for i, event in enumerate(outputs) if event.transcript == "好的，请稍等。"
    )
    started_index = next(
        i
        for i, event in enumerate(outputs)
        if event.task_event and event.task_event["event_type"] == "task.started"
    )
    assert ack_index < started_index
    completed_index = next(
        i
        for i, event in enumerate(outputs)
        if event.task_event and event.task_event["event_type"] == "task.completed"
    )
    final_voice_index = next(
        i for i, event in enumerate(outputs) if event.transcript == "查好了，资料已经整理好。"
    )
    assert completed_index < final_voice_index


@pytest.mark.asyncio
async def test_persona_agent_wait_tool_suppresses_output(tmp_path):
    plugin = await make_persona("wait_for_more_input", tmp_path / "wait.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                one_input(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert len(outputs) == 1
    assert outputs[0].user_transcript == "我觉得..."
    assert plugin.task_runtime.calls == []


def test_persona_event_logs_keep_stream_deltas_out_of_info(caplog):
    logger_name = "inference.plugins.voice_llm.persona_agent"

    with caplog.at_level(logging.INFO, logger=logger_name):
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(transcript="收"),
        )
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(
                transcript="收到",
                audio=AudioChunk(data=b"", sample_rate=24000, is_final=True),
                is_final=True,
            ),
        )
        PersonaAgentPlugin._log_model_event(
            "session-1",
            VoiceLLMOutputEvent(audio=AudioChunk(data=b"pcm", sample_rate=24000)),
        )

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    assert "kind=assistant_final" in messages[0]
    assert "收到" in messages[0]


@pytest.mark.asyncio
async def test_persona_agent_merges_waited_partial_into_create_task(tmp_path):
    plugin = await make_persona("partial_create_task", tmp_path / "partial.db")

    try:
        outputs = [
            event
            async for event in plugin.converse_stream(
                two_inputs(),
                VoiceLLMSessionConfig(session_id="session-1"),
            )
        ]
    finally:
        await plugin.shutdown()

    assert outputs[0].user_transcript == "我想让你帮我查一下"
    assert outputs[1].user_transcript == "今天知乎有哪些热门信息"
    assert outputs[2].transcript == "好的，请稍等。"
    assert outputs[-1].transcript == "查好了，资料已经整理好。"
    assert plugin.task_runtime.calls[0] == (
        "create_task",
        "session-1",
        {
            "description": "我想让你帮我查一下今天知乎有哪些热门信息",
            "user_request": "我想让你帮我查一下今天知乎有哪些热门信息",
        },
    )
