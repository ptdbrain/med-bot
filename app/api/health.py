from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request) -> dict:
    service = request.app.state.rag_service
    return {
        "status": "ok",
        "service": "medtutor-rag-copilot",
        "chunks_indexed": service.indexed_chunk_count,
        "vectors_indexed": service.vector_indexed_count,
        "llm_provider": service.settings.llm_provider,
        "embedding_provider": service.settings.embedding_provider,
        "vector_store_provider": service.settings.vector_store_provider,
        "reranker_provider": service.settings.reranker_provider,
    }
