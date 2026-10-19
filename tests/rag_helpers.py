from app.schemas.document import Chunk, ChunkMetadata, Document, SearchResult


def make_document(text: str) -> Document:
    return Document("test-doc", "", "test.md", "md", text, {"created_at": "2026-09-22T00:00:00Z"})


def make_result(chunk_id: str, text: str, signals: dict[str, float] | None = None) -> SearchResult:
    metadata = ChunkMetadata("test-doc", "test.md", "md", "", "2026-09-22T00:00:00Z", chunk_id)
    return SearchResult(Chunk(chunk_id, text, metadata), 0.8, signals or {})
