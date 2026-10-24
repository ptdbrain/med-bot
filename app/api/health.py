from __future__ import annotations

from fastapi import APIRouter, Request, Response

from app.runtime.doctor import check_runtime, is_ready

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


@router.get("/ready")
def readiness(request: Request, response: Response) -> dict:
    service = request.app.state.rag_service
    checks = check_runtime(service.settings)
    ready = is_ready(checks)
    response.status_code = 200 if ready else 503
    return {
        "status": "ready" if ready else "not_ready",
        "checks": [{"name": check.name, "status": check.status, "message": check.message} for check in checks],
    }
