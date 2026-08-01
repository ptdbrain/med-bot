from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.retrieval import RetrievalResponse, RetrievedChunk

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    result = request.app.state.rag_service.chat(
        question=payload.question,
        rag_mode=payload.rag_mode,
        top_k=payload.top_k,
    )
    return ChatResponse(**result.__dict__)


@router.get("/retrieve", response_model=RetrievalResponse)
def retrieve(query: str, request: Request, top_k: int = 5, rag_mode: str = "naive") -> RetrievalResponse:
    try:
        results = request.app.state.rag_service.retrieve(
            query,
            top_k=top_k,
            rag_mode=rag_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RetrievalResponse(
        query=query,
        mode=rag_mode,
        results=[
            RetrievedChunk(
                chunk_id=result.chunk.chunk_id,
                text=result.chunk.text,
                score=result.score,
                metadata=result.chunk.metadata.to_dict(),
                signals=result.signals,
            )
            for result in results
        ],
    )
