from __future__ import annotations

import math
from collections import Counter

from app.retrieval.vector_store import tokenize
from app.schemas.document import Chunk, SearchResult


class BM25Store:
    """Small dependency-free Okapi BM25 store for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: dict[str, list[str]] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        known = {chunk.chunk_id for chunk in self._chunks}
        for chunk in chunks:
            if chunk.chunk_id in known:
                continue
            self._chunks.append(chunk)
            self._tokens[chunk.chunk_id] = tokenize(chunk.text)
            known.add(chunk.chunk_id)

    def replace_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks = []
        self._tokens = {}
        self.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms or not self._chunks:
            return []

        document_frequency: Counter[str] = Counter()
        lengths: list[int] = []
        for chunk in self._chunks:
            tokens = self._tokens[chunk.chunk_id]
            document_frequency.update(set(tokens))
            lengths.append(len(tokens))

        total_documents = len(self._chunks)
        average_length = sum(lengths) / max(1, total_documents)
        scored: list[SearchResult] = []
        for chunk in self._chunks:
            tokens = self._tokens[chunk.chunk_id]
            counts = Counter(tokens)
            document_length = len(tokens)
            score = 0.0
            for term in query_terms:
                term_frequency = counts.get(term, 0)
                if term_frequency == 0:
                    continue
                doc_frequency = document_frequency[term]
                inverse_document_frequency = math.log(
                    1 + (total_documents - doc_frequency + 0.5) / (doc_frequency + 0.5)
                )
                length_normalization = self.k1 * (
                    1 - self.b + self.b * document_length / max(1.0, average_length)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1) / (term_frequency + length_normalization)
                )
            if score > 0:
                scored.append(SearchResult(chunk=chunk, score=round(score, 6)))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
