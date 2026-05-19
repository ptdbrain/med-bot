from __future__ import annotations

from collections import Counter

from app.retrieval.vector_store import tokenize
from app.schemas.document import Chunk, SearchResult


class BM25Store:
    """Simple lexical store reserved for Phase 2 hybrid search."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            counts = Counter(tokenize(chunk.text))
            score = sum(counts.get(term, 0) for term in query_terms)
            if score:
                scored.append(SearchResult(chunk=chunk, score=float(score)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
