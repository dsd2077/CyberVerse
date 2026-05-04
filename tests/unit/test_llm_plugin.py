import pytest

from inference.plugins.llm.openai_plugin import OpenAILLMPlugin


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _EmptyStream()


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


@pytest.mark.asyncio
async def test_openai_plugin_does_not_prepend_default_when_system_message_exists():
    plugin = OpenAILLMPlugin()
    plugin.client = _FakeClient()
    plugin.system_prompt = "plugin default"

    chunks = [
        chunk
        async for chunk in plugin.generate_stream(
            [
                {"role": "system", "content": "request system"},
                {"role": "user", "content": "hello"},
            ]
        )
    ]

    assert chunks[-1].is_final is True
    messages = plugin.client.chat.completions.kwargs["messages"]
    assert [message["content"] for message in messages if message["role"] == "system"] == [
        "request system"
    ]
