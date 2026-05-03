import grpc

from inference.core.registry import PluginRegistry
from inference.generated import llm_pb2, llm_pb2_grpc
from inference.plugins.llm.base import LLMPlugin


class LLMGRPCService(llm_pb2_grpc.LLMServiceServicer):

    def __init__(self, registry: PluginRegistry) -> None:
        self.registry = registry

    def _get_plugin(self, provider: str = "") -> LLMPlugin:
        provider = provider.strip()
        if provider:
            return self.registry.get(f"llm.{provider}")
        plugin = self.registry.get_by_category("llm")
        if plugin is None:
            raise RuntimeError("No LLM plugin initialized")
        return plugin

    async def GenerateStream(self, request, context):
        provider = request.config.provider if request.config else ""
        try:
            plugin = self._get_plugin(provider)
        except (KeyError, RuntimeError) as exc:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))

        messages = [
            {"role": msg.role, "content": msg.content}
            for msg in request.messages
        ]
        async for chunk in plugin.generate_stream(messages):
            yield llm_pb2.LLMChunk(
                token=chunk.token,
                accumulated_text=chunk.accumulated_text,
                is_sentence_end=chunk.is_sentence_end,
                is_final=chunk.is_final,
            )
