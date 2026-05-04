import base64
import os
from typing import AsyncIterator

from inference.core.types import LLMResponseChunk, PluginConfig
from inference.plugins.llm.base import LLMPlugin

SENTENCE_ENDERS = {"。", "！", "？", ".", "!", "?", "；", ";", "\n"}


class OpenAILLMPlugin(LLMPlugin):
    name = "llm.openai"
    supports_images = True

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

    @staticmethod
    def _format_message(message: dict) -> dict:
        images = message.get("images") or []
        content = message.get("content", "")
        if not images:
            return {"role": message["role"], "content": content}

        parts = []
        if content:
            parts.append({"type": "text", "text": content})
        for image in images:
            raw = image.get("data", b"")
            if isinstance(raw, str):
                encoded = raw
            else:
                encoded = base64.b64encode(raw).decode("ascii")
            mime_type = image.get("mime_type") or "image/jpeg"
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
                }
            )
        return {"role": message["role"], "content": parts}

    async def generate_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[LLMResponseChunk]:
        full_messages = [self._format_message(message) for message in messages]
        if self.system_prompt:
            full_messages = [{"role": "system", "content": self.system_prompt}] + full_messages

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
