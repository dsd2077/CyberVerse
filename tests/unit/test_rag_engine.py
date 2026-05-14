import pytest

from inference.rag.engine import HashEmbeddings, RAGEngine, RAGIndexRequest, RAGSearchRequest


def test_hash_embeddings_are_deterministic():
    embeddings = HashEmbeddings(dimensions=32)

    first = embeddings.embed_query("角色出生在海边")
    second = embeddings.embed_query("角色出生在海边")

    assert first == second
    assert len(first) == 32


def test_rag_engine_resolves_server_data_path_from_shared_data_dir(tmp_path, monkeypatch):
    shared_data = tmp_path / "data"
    source = shared_data / "characters" / "角色_a1" / "knowledge" / "sources" / "requirement.txt"
    source.parent.mkdir(parents=True)
    source.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("CYBERVERSE_DATA_DIR", str(shared_data))

    engine = RAGEngine({})
    resolved = engine._resolve_source_path(
        "/root/data/characters/角色_a1/knowledge/sources/requirement.txt"
    )

    assert resolved == source.resolve()


@pytest.mark.asyncio
async def test_rag_engine_indexes_searches_and_deletes_text_source(tmp_path):
    pytest.importorskip("langchain_chroma")
    pytest.importorskip("langchain_community")
    pytest.importorskip("langchain_text_splitters")

    source_path = tmp_path / "source.txt"
    source_path.write_text("晴天出生在海边，后来成为工程师。", encoding="utf-8")
    engine = RAGEngine(
        {
            "pipeline": {
                "rag": {
                    "chunk_chars": 120,
                    "chunk_overlap_chars": 0,
                    "top_k": 3,
                    "max_context_chars": 500,
                    "min_score": 0.0,
                }
            },
            "inference": {"embedding": {"default": "fake", "fake": {"dimensions": 64}}},
        }
    )
    request = RAGIndexRequest(
        character_id="char_1",
        character_dir=str(tmp_path / "character"),
        source_id="source_1",
        source_type="",
        title="profile",
        filename="source.txt",
        mime_type="text/plain",
        source_path=str(source_path),
    )

    chunk_count = await engine.index_source(request)
    assert chunk_count >= 1

    results = await engine.search(
        RAGSearchRequest(
            character_id="char_1",
            character_dir=request.character_dir,
            query="晴天在哪里出生",
            top_k=3,
            min_score=0.0,
        )
    )
    assert results
    assert results[0].source_id == "source_1"
    assert "出生在海边" in results[0].content

    await engine.delete_source("char_1", request.character_dir, "source_1")
    results = await engine.search(
        RAGSearchRequest(
            character_id="char_1",
            character_dir=request.character_dir,
            query="晴天在哪里出生",
            top_k=3,
            min_score=0.0,
        )
    )
    assert results == []
