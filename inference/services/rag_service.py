import grpc

from inference.core.config import load_config
from inference.generated import rag_pb2, rag_pb2_grpc
from inference.rag import RAGEngine, RAGIndexRequest, RAGSearchRequest


class RAGGRPCService(rag_pb2_grpc.RAGServiceServicer):
    def __init__(self, config: dict | str | None = None) -> None:
        if isinstance(config, str):
            config = load_config(config)
        self.engine = RAGEngine(config or {})

    async def IndexSource(self, request, context):
        try:
            count = await self.engine.index_source(
                RAGIndexRequest(
                    character_id=request.character_id,
                    character_dir=request.character_dir,
                    source_id=request.source_id,
                    source_type=request.source_type,
                    title=request.title,
                    filename=request.filename,
                    mime_type=request.mime_type,
                    source_path=request.source_path,
                )
            )
            return rag_pb2.RAGIndexSourceResponse(chunk_count=count)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def DeleteSource(self, request, context):
        try:
            await self.engine.delete_source(
                request.character_id,
                request.character_dir,
                request.source_id,
            )
            return rag_pb2.RAGDeleteSourceResponse(success=True)
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))

    async def Search(self, request, context):
        try:
            results = await self.engine.search(
                RAGSearchRequest(
                    character_id=request.character_id,
                    character_dir=request.character_dir,
                    query=request.query,
                    top_k=request.top_k,
                    max_context_chars=request.max_context_chars,
                    min_score=request.min_score,
                )
            )
            return rag_pb2.RAGSearchResponse(
                results=[
                    rag_pb2.RAGSearchResult(
                        source_id=item.source_id,
                        source_type=item.source_type,
                        title=item.title,
                        filename=item.filename,
                        content=item.content,
                        score=item.score,
                    )
                    for item in results
                ]
            )
        except Exception as exc:
            await context.abort(grpc.StatusCode.INTERNAL, str(exc))
