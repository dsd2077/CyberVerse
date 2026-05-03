from inference.core.types import PluginConfig
from inference.plugins.llm.openai_plugin import OpenAILLMPlugin
from inference.plugins.qwen_endpoint import dashscope_base_url


class QwenLLMPlugin(OpenAILLMPlugin):
    """DashScope Qwen chat-completions plugin using the OpenAI-compatible API."""

    name = "llm.qwen"

    async def initialize(self, config: PluginConfig) -> None:
        params = {
            "model": "qwen3.6-plus",
            "temperature": 0.7,
            "extra_body": {"enable_thinking": False},
            **config.params,
        }
        params["base_url"] = dashscope_base_url()
        if not params.get("extra_body"):
            params["extra_body"] = {"enable_thinking": False}
        await super().initialize(
            PluginConfig(
                plugin_name=config.plugin_name,
                params=params,
                shared=config.shared,
            )
        )
