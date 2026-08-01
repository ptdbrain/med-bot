from __future__ import annotations

from app.config import Settings
from app.retrieval.reranker import (
    CrossEncoderReranker,
    LexicalReranker,
    Reranker,
)


def create_reranker(settings: Settings) -> Reranker:
    provider = settings.reranker_provider.strip().lower()
    if provider in {"lexical", "local"}:
        return LexicalReranker(
            diversity_lambda=settings.reranker_diversity_lambda,
        )
    if provider in {"cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker(
            model_name=settings.reranker_model,
            device=settings.embedding_device,
            batch_size=settings.reranker_batch_size,
            diversity_lambda=settings.reranker_diversity_lambda,
        )
    raise ValueError(
        f"RERANKER_PROVIDER='{settings.reranker_provider}' is invalid. " "Supported providers: lexical, cross-encoder."
    )
