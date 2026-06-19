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
    }
