from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    doc_id: str
    title: str
    source_path: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    doc_id: str
    source_path: str
    source_type: str
    title: str
    created_at: str
    hash: str
    page_number: int | None = None
    section_title: str | None = None
    image_path: str | None = None
    parent_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            metadata=ChunkMetadata(**data["metadata"]),
        )


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "score": self.score,
            "signals": self.signals,
        }
