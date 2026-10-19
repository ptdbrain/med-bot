from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    rag_mode: str = "naive"
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    chunk_id: str
    title: str
    source_path: str
    source_type: str
    score: float
    page_number: int | None = None
    section_title: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    retrieved_chunks: list[dict] = Field(default_factory=list)
    mode_used: str = "naive"
    safety_warning: str | None = None
    validation: dict | None = None
