from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from app.schemas.document import Chunk, SearchResult
from app.utils.file_utils import ensure_directory


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def cosine_score(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[token] * right.get(token, 0) for token in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class InMemoryVectorStore:
    """Local lexical vector fallback for offline demo and tests.

    The class intentionally keeps the interface narrow so a Qdrant-backed
    implementation can replace it without changing RAG orchestration.
    """

    def __init__(self, persist_path: Path | None = None) -> None:
        self.persist_path = persist_path
        self._chunks: list[Chunk] = []
        self._vectors: dict[str, Counter[str]] = {}
        if persist_path and persist_path.exists():
            self.load()

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        known = {chunk.chunk_id for chunk in self._chunks}
        for chunk in chunks:
            if chunk.chunk_id in known:
                continue
            self._chunks.append(chunk)
            self._vectors[chunk.chunk_id] = Counter(tokenize(chunk.text))
        self.save()

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        query_vector = Counter(tokenize(query))
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            score = cosine_score(query_vector, self._vectors.get(chunk.chunk_id, Counter()))
            if score > min_score:
                scored.append(SearchResult(chunk=chunk, score=round(score, 6)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def save(self) -> None:
        if not self.persist_path:
            return
        ensure_directory(self.persist_path.parent)
        payload = [chunk.to_dict() for chunk in self._chunks]
        self.persist_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        payload = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.from_dict(item) for item in payload]
        self._vectors = {chunk.chunk_id: Counter(tokenize(chunk.text)) for chunk in self._chunks}


class QdrantVectorStore:
    """Future Qdrant adapter boundary.

    Phase 1 uses InMemoryVectorStore. This adapter exists so later work can add
    Qdrant without changing API, ingestion, or RAG modules.
    """

    def __init__(self, url: str) -> None:
        self.url = url

    def add_chunks(self, chunks: list[Chunk]) -> None:
        raise NotImplementedError("Qdrant storage is planned after the Phase 1 local fallback.")

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        raise NotImplementedError("Qdrant search is planned after the Phase 1 local fallback.")
