from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.runtime.doctor import check_runtime

router = APIRouter(tags=["health"])


@router.get("/ready")
def ready(request: Request) -> JSONResponse:
    checks = check_runtime(request.app.state.rag_service.settings, probe_models=True)
    is_ready = all(check.status == "ok" for check in checks)
    return JSONResponse(
        {"ready": is_ready, "checks": [asdict(c) for c in checks]}, status_code=200 if is_ready else 503
    )


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
