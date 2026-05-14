import pytest

from inference.core.types import ToolCall, VoiceLLMSessionConfig
from inference.plugins.voice_llm.persona_agent import PersonaAgentPlugin
from inference.rag import RAGSearchResult


class FakeRAGEngine:
    def __init__(self):
        self.requests = []

    async def search(self, request):
        self.requests.append(request)
        return [
            RAGSearchResult(
                source_id="s1",
                source_type="",
                title="profile",
                filename="bio.txt",
                content="晴天出生在海边。",
                score=0.91,
            )
        ]


@pytest.mark.asyncio
async def test_persona_rag_tool_returns_empty_without_character_dir():
    plugin = PersonaAgentPlugin()
    call = ToolCall(
        id="call_1",
        name="retrieve_character_knowledge",
        arguments={"query": "早年经历"},
    )

    result = await plugin._retrieve_character_knowledge(
        call,
        VoiceLLMSessionConfig(session_id="s1", character_id="c1"),
    )

    assert result.result["ok"] is True
    assert result.result["results"] == []
    assert result.result["reason"] == "character_dir_missing"


@pytest.mark.asyncio
async def test_persona_rag_tool_searches_with_character_context():
    plugin = PersonaAgentPlugin()
    plugin.rag_engine = FakeRAGEngine()
    call = ToolCall(
        id="call_1",
        name="retrieve_character_knowledge",
        arguments={"query": "早年经历"},
    )

    result = await plugin._retrieve_character_knowledge(
        call,
        VoiceLLMSessionConfig(session_id="s1", character_id="c1", character_dir="/tmp/character"),
    )

    assert result.result["ok"] is True
    assert result.result["query"] == "早年经历"
    assert result.result["results"][0]["content"] == "晴天出生在海边。"
    assert "source_type" not in result.result["results"][0]
    assert plugin.rag_engine.requests[0].character_id == "c1"
    assert plugin.rag_engine.requests[0].character_dir == "/tmp/character"


@pytest.mark.asyncio
async def test_persona_rag_pre_response_returns_instructions_for_hits():
    plugin = PersonaAgentPlugin()
    plugin.rag_engine = FakeRAGEngine()

    instructions = await plugin._rag_response_instructions(
        "看山呀，你为什么姓刘？",
        VoiceLLMSessionConfig(
            session_id="s1",
            character_id="c1",
            character_dir="/tmp/character",
            bot_name="刘看山",
        ),
    )

    assert "【角色素材检索结果】" in instructions
    assert "晴天出生在海边。" in instructions
    assert plugin.rag_engine.requests[0].query == "看山呀，你为什么姓刘？"
