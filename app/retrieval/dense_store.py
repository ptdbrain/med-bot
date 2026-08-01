from __future__ import annotations

import json
import math
import sqlite3
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.embeddings.base import EmbeddingProvider
from app.retrieval.vector_store import chunk_search_text
from app.schemas.document import Chunk, SearchResult
from app.utils.file_utils import ensure_directory


class SQLiteDenseVectorStore:
    """Persistent local dense store for development without a Qdrant server."""

    def __init__(
        self,
        path: Path,
        embedding_provider: EmbeddingProvider,
        vector_size: int,
        write_batch_size: int = 64,
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")
        if write_batch_size <= 0:
            raise ValueError("write_batch_size must be positive")
        self.path = path
        self.embedding_provider = embedding_provider
        self.vector_size = vector_size
        self.write_batch_size = write_batch_size
        ensure_directory(path.parent)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS dense_vectors (
                    chunk_id TEXT PRIMARY KEY,
                    vector BLOB NOT NULL,
                    chunk_json TEXT NOT NULL
                )
                """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS dense_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """)
            row = connection.execute("SELECT value FROM dense_metadata WHERE key = 'vector_size'").fetchone()
            if row is not None and int(row[0]) != self.vector_size:
                raise ValueError(
                    "Dense index dimension does not match EMBEDDING_DIMENSION. " "Clear or rebuild the dense index."
                )
            connection.execute(
                """
                INSERT OR REPLACE INTO dense_metadata(key, value)
                VALUES ('vector_size', ?)
                """,
                (str(self.vector_size),),
            )

    @property
    def chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_json FROM dense_vectors ORDER BY rowid").fetchall()
        return [Chunk.from_dict(json.loads(str(row[0]))) for row in rows]

    @property
    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM dense_vectors").fetchone()
        return int(row[0])

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        for start in range(0, len(chunks), self.write_batch_size):
            batch = chunks[start : start + self.write_batch_size]
            vectors = self.embedding_provider.embed_documents([chunk_search_text(chunk) for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError("Embedding provider returned a different number of " "vectors than chunks.")

            rows: list[tuple[str, bytes, str]] = []
            for chunk, vector in zip(batch, vectors, strict=True):
                rows.append(
                    (
                        chunk.chunk_id,
                        self._pack_vector(vector),
                        json.dumps(
                            chunk.to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                )
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO dense_vectors(chunk_id, vector, chunk_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET
                        vector = excluded.vector,
                        chunk_json = excluded.chunk_json
                    """,
                    rows,
                )

    def existing_chunk_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id FROM dense_vectors").fetchall()
        return {str(row[0]) for row in rows}

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        with self._connect() as connection:
            connection.executemany(
                "DELETE FROM dense_vectors WHERE chunk_id = ?",
                [(chunk_id,) for chunk_id in chunk_ids],
            )

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM dense_vectors")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        if top_k <= 0:
            return []
        query_vector = self.embedding_provider.embed_query(query)
        if len(query_vector) != self.vector_size:
            raise RuntimeError("Query embedding dimension does not match the dense index.")
        query_norm = math.sqrt(sum(float(value) * float(value) for value in query_vector))
        if query_norm == 0:
            return []

        with self._connect() as connection:
            rows = connection.execute("SELECT chunk_id, vector, chunk_json FROM dense_vectors").fetchall()

        scored_payloads: list[tuple[float, str, str]] = []
        for chunk_id, vector_blob, chunk_json in rows:
            vector = self._unpack_vector(vector_blob)
            vector_norm = math.sqrt(sum(value * value for value in vector))
            if vector_norm == 0:
                continue
            dot_product = sum(float(left) * right for left, right in zip(query_vector, vector, strict=True))
            score = dot_product / (query_norm * vector_norm)
            if score > min_score:
                scored_payloads.append((round(score, 6), str(chunk_id), str(chunk_json)))
        scored_payloads.sort(key=lambda item: (-item[0], item[1]))
        return [
            SearchResult(
                chunk=Chunk.from_dict(json.loads(chunk_json)),
                score=score,
                signals={"semantic_score": score},
            )
            for score, _, chunk_json in scored_payloads[:top_k]
        ]

    def _pack_vector(self, vector: list[float]) -> bytes:
        if len(vector) != self.vector_size:
            raise RuntimeError("Document embedding dimension does not match " "EMBEDDING_DIMENSION.")
        return struct.pack(
            f"<{self.vector_size}f",
            *(float(value) for value in vector),
        )

    def _unpack_vector(self, payload: bytes) -> tuple[float, ...]:
        expected_size = self.vector_size * 4
        if len(payload) != expected_size:
            raise RuntimeError("Stored dense vector has an invalid size.")
        return struct.unpack(f"<{self.vector_size}f", payload)
