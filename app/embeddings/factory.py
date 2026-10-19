from __future__ import annotations

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformer import SentenceTransformerEmbedding


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider | None:
    provider = settings.embedding_provider.strip().lower()
    if provider == "http":
        from app.embeddings.http import HTTPEmbedding

        return HTTPEmbedding(settings)
    if provider in {"lexical", "local"}:
        return None
    if provider == "sentence-transformers":
        return SentenceTransformerEmbedding(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        )
    raise ValueError(
        f"EMBEDDING_PROVIDER='{settings.embedding_provider}' không hợp lệ. " "Hỗ trợ: lexical, sentence-transformers."
    )
