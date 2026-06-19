from __future__ import annotations

from app.retrieval.vector_store import tokenize
from app.schemas.document import SearchResult


class LexicalReranker:
    """Deterministic local reranker used by the basic rerank pipeline."""

    def rerank(self, query: str, results: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        rescored: list[SearchResult] = []
        for result in results:
            chunk_terms = set(tokenize(result.chunk.text))
            overlap = len(query_terms & chunk_terms)
            score = result.score + (overlap * 0.05)
            rescored.append(SearchResult(chunk=result.chunk, score=round(score, 6)))
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k]
