import os
from typing import AsyncIterator

from inference.core.types import LLMResponseChunk, PluginConfig
from inference.plugins.llm.base import LLMPlugin

SENTENCE_ENDERS = {"。", "！", "？", ".", "!", "?", "；", ";", "\n"}


class OpenAILLMPlugin(LLMPlugin):
    name = "llm.openai"

    def __init__(self) -> None:
        self.client = None
        self.model = "gpt-4o"
        self.temperature = 0.7
        self.system_prompt = ""
        self.extra_body: dict = {}

    async def initialize(self, config: PluginConfig) -> None:
        from openai import AsyncOpenAI

        env_base_url = (
            os.environ.get("OPENAI_BASE_URL")
            if config.plugin_name == "llm.openai"
            else ""
        )
        base_url = env_base_url or config.params.get("base_url")
        client_kwargs = {"api_key": config.params.get("api_key")}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = AsyncOpenAI(**client_kwargs)
        self.model = config.params.get("model", "gpt-4o")
        self.temperature = float(config.params.get("temperature", 0.7))
        self.system_prompt = config.params.get("system_prompt", "")
        extra_body = config.params.get("extra_body", {})
        self.extra_body = extra_body if isinstance(extra_body, dict) else {}

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMResponseChunk]:
        full_messages = messages
        if self.system_prompt:
            full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        accumulated = ""
        create_kwargs = {
            "model": self.model,
            "messages": full_messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.extra_body:
            create_kwargs["extra_body"] = self.extra_body
        stream = await self.client.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                accumulated += token
                is_sentence_end = any(token.endswith(p) for p in SENTENCE_ENDERS)
                yield LLMResponseChunk(
                    token=token,
                    accumulated_text=accumulated,
                    is_sentence_end=is_sentence_end,
                    is_final=False,
                )

        yield LLMResponseChunk(
            token="",
            accumulated_text=accumulated,
            is_sentence_end=True,
            is_final=True,
        )

    async def shutdown(self) -> None:
        self.client = None
