from __future__ import annotations

import json
import math
import re
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

import requests

from app.embeddings.base import EmbeddingProvider
from app.schemas.document import Chunk, SearchResult
from app.utils.file_utils import ensure_directory

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[Chunk]) -> None: ...

    def delete_chunks(self, chunk_ids: list[str]) -> None: ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]: ...


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


def chunk_search_text(chunk: Chunk) -> str:
    metadata = chunk.metadata
    return "\n".join(
        value
        for value in (
            metadata.title,
            metadata.section_title,
            chunk.text,
        )
        if value
    )


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

    @property
    def count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        positions = {chunk.chunk_id: index for index, chunk in enumerate(self._chunks)}
        for chunk in chunks:
            position = positions.get(chunk.chunk_id)
            if position is None:
                positions[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
            else:
                self._chunks[position] = chunk
            self._vectors[chunk.chunk_id] = Counter(tokenize(chunk_search_text(chunk)))
        self.save()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        deleted = set(chunk_ids)
        self._chunks = [chunk for chunk in self._chunks if chunk.chunk_id not in deleted]
        for chunk_id in deleted:
            self._vectors.pop(chunk_id, None)
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
        self._vectors = {chunk.chunk_id: Counter(tokenize(chunk_search_text(chunk))) for chunk in self._chunks}


class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        embedding_provider: EmbeddingProvider,
        vector_size: int,
        session=None,
        timeout_seconds: int = 30,
    ) -> None:
        self.url = url.rstrip("/")
        self.collection = collection
        self.embedding_provider = embedding_provider
        self.vector_size = vector_size
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self._ensure_collection()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedding_provider.embed_documents([chunk_search_text(chunk) for chunk in chunks])
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider trả về số vector không khớp số chunk.")
        points = [
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                "vector": vector,
                "payload": {"chunk": chunk.to_dict()},
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        response = self.session.put(
            f"{self._collection_url()}/points",
            params={"wait": "true"},
            json={"points": points},
            timeout=self.timeout_seconds,
        )
        self._require_success(response, "upsert chunks")

    def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[SearchResult]:
        vector = self.embedding_provider.embed_query(query)
        response = self.session.post(
            f"{self._collection_url()}/points/search",
            json={
                "vector": vector,
                "limit": top_k,
                "with_payload": True,
                "score_threshold": min_score,
            },
            timeout=self.timeout_seconds,
        )
        self._require_success(response, "search")
        results: list[SearchResult] = []
        for item in response.json().get("result", []):
            chunk_data = item.get("payload", {}).get("chunk")
            if not chunk_data:
                continue
            results.append(
                SearchResult(
                    chunk=Chunk.from_dict(chunk_data),
                    score=float(item.get("score", 0.0)),
                    signals={
                        "semantic_score": float(item.get("score", 0.0)),
                    },
                )
            )
        return results

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id)) for chunk_id in chunk_ids]
        response = self.session.post(
            f"{self._collection_url()}/points/delete",
            params={"wait": True},
            json={"points": point_ids},
            timeout=self.timeout_seconds,
        )
        self._require_success(response, "delete chunks")

    def clear(self) -> None:
        response = self.session.delete(
            self._collection_url(),
            timeout=self.timeout_seconds,
        )
        if response.status_code not in {200, 404}:
            self._require_success(response, "delete collection")
        self._ensure_collection()

    def _collection_url(self) -> str:
        collection = quote(self.collection, safe="")
        return f"{self.url}/collections/{collection}"

    def _ensure_collection(self) -> None:
        response = self.session.get(
            self._collection_url(),
            timeout=self.timeout_seconds,
        )
        if response.status_code == 200:
            return
        if response.status_code != 404:
            self._require_success(response, "check collection")
        response = self.session.put(
            self._collection_url(),
            json={
                "vectors": {
                    "size": self.vector_size,
                    "distance": "Cosine",
                }
            },
            timeout=self.timeout_seconds,
        )
        self._require_success(response, "create collection")

    def _require_success(self, response, operation: str) -> None:
        if 200 <= response.status_code < 300:
            return
        raise RuntimeError(
            f"Qdrant {operation} thất bại tại {self.url}: " f"HTTP {response.status_code} {response.text}"
        )
