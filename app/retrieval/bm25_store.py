from __future__ import annotations

import math
from collections import Counter

from app.retrieval.vector_store import chunk_search_text, tokenize
from app.schemas.document import Chunk, SearchResult


class BM25Store:
    """Small dependency-free Okapi BM25 store for hybrid retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tokens: dict[str, list[str]] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._lengths: dict[str, int] = {}
        self._by_id: dict[str, Chunk] = {}
        self._average_length = 0.0

    def add_chunks(self, chunks: list[Chunk]) -> None:
        known = {chunk.chunk_id for chunk in self._chunks}
        for chunk in chunks:
            if chunk.chunk_id in known:
                continue
            self._chunks.append(chunk)
            self._tokens[chunk.chunk_id] = tokenize(chunk_search_text(chunk))
            known.add(chunk.chunk_id)
        self._rebuild_statistics()

    def _rebuild_statistics(self):
        self._postings = {}
        self._by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        self._lengths = {key: len(tokens) for key, tokens in self._tokens.items()}
        self._average_length = sum(self._lengths.values()) / max(1, len(self._lengths))
        for chunk_id, tokens in self._tokens.items():
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, {})[chunk_id] = count

    def replace_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks = []
        self._tokens = {}
        self.add_chunks(chunks)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_terms = set(tokenize(query))
        if not query_terms or not self._chunks or top_k <= 0:
            return []
        total_documents = len(self._chunks)
        scores: Counter[str] = Counter()
        for term in query_terms:
            postings = self._postings.get(term, {})
            frequency = len(postings)
            idf = math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))
            for chunk_id, tf in postings.items():
                norm = self.k1 * (1 - self.b + self.b * self._lengths[chunk_id] / max(1.0, self._average_length))
                scores[chunk_id] += idf * tf * (self.k1 + 1) / (tf + norm)
        return [
            SearchResult(self._by_id[key], round(value, 6))
            for key, value in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        ]
