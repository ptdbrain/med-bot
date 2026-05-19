from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    metadata: dict


class RetrievalResponse(BaseModel):
    query: str
    mode: str = "naive"
    results: list[RetrievedChunk] = Field(default_factory=list)
