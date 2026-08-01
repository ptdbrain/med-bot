from __future__ import annotations

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.retrieval.dense_store import SQLiteDenseVectorStore
from app.retrieval.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStore,
)


def create_vector_store(
    settings: Settings,
    embedding_provider: EmbeddingProvider | None,
) -> VectorStore:
    provider = settings.vector_store_provider.strip().lower()
    if provider in {"memory", "local"}:
        return InMemoryVectorStore(settings.index_path)
    if provider in {"sqlite-dense", "dense-local"}:
        if embedding_provider is None:
            raise ValueError(
                "sqlite-dense requires a dense embedding provider. " "Set EMBEDDING_PROVIDER=sentence-transformers."
            )
        return SQLiteDenseVectorStore(
            path=settings.dense_index_path,
            embedding_provider=embedding_provider,
            vector_size=settings.embedding_dimension,
            write_batch_size=settings.dense_write_batch_size,
        )
    if provider == "qdrant":
        if embedding_provider is None:
            raise ValueError("Qdrant cần dense embedding provider. " "Đặt EMBEDDING_PROVIDER=sentence-transformers.")
        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            embedding_provider=embedding_provider,
            vector_size=settings.embedding_dimension,
        )
    raise ValueError(
        f"VECTOR_STORE_PROVIDER='{settings.vector_store_provider}' không hợp lệ. "
        "Hỗ trợ: memory, sqlite-dense, qdrant."
    )
