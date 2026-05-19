from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.document import Chunk, ChunkMetadata


def test_local_vector_store_returns_relevant_chunks():
    store = InMemoryVectorStore()
    store.add_chunks(
        [
            Chunk(
                chunk_id="stroke-1",
                text="Dot quy co dau hieu FAST, meo mieng, yeu tay, noi kho.",
                metadata=ChunkMetadata(
                    doc_id="stroke",
                    source_path="data/sample/stroke.md",
                    source_type="md",
                    title="Dot quy",
                    created_at="2026-05-19T00:00:00Z",
                    hash="abc",
                ),
            ),
            Chunk(
                chunk_id="lab-1",
                text="Hemoglobin la chi so xet nghiem mau.",
                metadata=ChunkMetadata(
                    doc_id="lab",
                    source_path="data/sample/lab.md",
                    source_type="md",
                    title="Xet nghiem mau",
                    created_at="2026-05-19T00:00:00Z",
                    hash="def",
                ),
            ),
        ]
    )

    results = store.search("dau hieu dot quy", top_k=1)

    assert results[0].chunk.chunk_id == "stroke-1"
    assert results[0].score > 0
