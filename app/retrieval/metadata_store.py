from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.schemas.document import Chunk, ChunkMetadata, Document, SearchResult
from app.utils.file_utils import ensure_directory
from app.utils.hashing import short_hash


class SQLiteMetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        ensure_directory(db_path.parent)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_sections (
                    parent_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    section_title TEXT,
                    page_number INTEGER,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS url_sources (
                    url TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def upsert_document(self, document: Document) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO documents(doc_id, title, source_path, source_type, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.title,
                    document.source_path,
                    document.source_type,
                    json.dumps(document.metadata, ensure_ascii=False),
                ),
            )

    def upsert_chunks(self, chunks: list[Chunk]) -> None:
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks(chunk_id, doc_id, text, metadata_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.metadata.doc_id,
                        chunk.text,
                        json.dumps(chunk.metadata.to_dict(), ensure_ascii=False),
                    )
                    for chunk in chunks
                ],
            )

    def upsert_parent_sections(self, chunks: list[Chunk]) -> None:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            parent_id = chunk.metadata.parent_id
            if parent_id:
                grouped.setdefault(parent_id, []).append(chunk)
        if not grouped:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO parent_sections(
                    parent_id,
                    doc_id,
                    title,
                    section_title,
                    page_number,
                    text,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        parent_id,
                        section_chunks[0].metadata.doc_id,
                        section_chunks[0].metadata.title,
                        section_chunks[0].metadata.section_title,
                        section_chunks[0].metadata.page_number,
                        "\n\n".join(chunk.text for chunk in section_chunks),
                        json.dumps(
                            section_chunks[0].metadata.to_dict(),
                            ensure_ascii=False,
                        ),
                    )
                    for parent_id, section_chunks in grouped.items()
                ],
            )

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])

    def list_chunks(self) -> list[Chunk]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, text, metadata_json
                FROM chunks
                ORDER BY rowid
                """
            ).fetchall()
        return [
            Chunk(
                chunk_id=str(row[0]),
                text=str(row[1]),
                metadata=ChunkMetadata(**json.loads(row[2])),
            )
            for row in rows
        ]

    def list_parent_sections(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    parent_id,
                    doc_id,
                    title,
                    section_title,
                    page_number,
                    text,
                    metadata_json
                FROM parent_sections
                ORDER BY rowid
                """
            ).fetchall()
        return [
            {
                "parent_id": str(row[0]),
                "doc_id": str(row[1]),
                "title": str(row[2]),
                "section_title": row[3],
                "page_number": row[4],
                "text": str(row[5]),
                "metadata": json.loads(row[6]),
            }
            for row in rows
        ]

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    documents.doc_id,
                    documents.title,
                    documents.source_path,
                    documents.source_type,
                    documents.metadata_json,
                    COUNT(chunks.chunk_id)
                FROM documents
                LEFT JOIN chunks ON chunks.doc_id = documents.doc_id
                GROUP BY documents.doc_id
                ORDER BY documents.rowid DESC
                """
            ).fetchall()
        return [
            {
                "doc_id": str(row[0]),
                "title": str(row[1]),
                "source_path": str(row[2]),
                "source_type": str(row[3]),
                "metadata": json.loads(row[4]),
                "chunks_count": int(row[5]),
            }
            for row in rows
        ]

    def get_document(self, doc_id: str) -> dict | None:
        return next(
            (
                document
                for document in self.list_documents()
                if document["doc_id"] == doc_id
            ),
            None,
        )

    def get_url_source(self, url: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT url, content_hash, doc_id, updated_at
                FROM url_sources
                WHERE url = ?
                """,
                (url,),
            ).fetchone()
        if row is None:
            return None
        return {
            "url": str(row[0]),
            "content_hash": str(row[1]),
            "doc_id": str(row[2]),
            "updated_at": str(row[3]),
        }

    def upsert_url_source(
        self,
        url: str,
        content_hash: str,
        doc_id: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO url_sources(url, content_hash, doc_id, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(url) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    doc_id = excluded.doc_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (url, content_hash, doc_id),
            )

    def chunk_ids_for_document(self, doc_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY rowid",
                (doc_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                "DELETE FROM parent_sections WHERE doc_id = ?",
                (doc_id,),
            )
            connection.execute(
                "DELETE FROM chunks WHERE doc_id = ?",
                (doc_id,),
            )
            connection.execute(
                "DELETE FROM documents WHERE doc_id = ?",
                (doc_id,),
            )
            connection.execute(
                "DELETE FROM url_sources WHERE doc_id = ?",
                (doc_id,),
            )
        return True

    def expand_parent_results(
        self,
        child_results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        all_chunks = self.list_chunks()
        expanded: list[SearchResult] = []
        seen: set[str] = set()
        for child_result in child_results:
            child = child_result.chunk
            metadata = child.metadata
            parent_key = metadata.parent_id or (
                f"{metadata.doc_id}:{metadata.section_title or '__document__'}"
            )
            if parent_key in seen:
                continue
            seen.add(parent_key)
            stored_parent = self._get_parent_section(metadata.parent_id)
            if stored_parent:
                parent_chunk = stored_parent
            else:
                siblings = [
                    chunk
                    for chunk in all_chunks
                    if (
                        chunk.metadata.parent_id == metadata.parent_id
                        if metadata.parent_id
                        else chunk.metadata.doc_id == metadata.doc_id
                        and chunk.metadata.section_title == metadata.section_title
                    )
                ]
                if not siblings:
                    siblings = [child]
                parent_metadata = ChunkMetadata(
                    **{
                        **metadata.to_dict(),
                        "parent_id": parent_key,
                    }
                )
                parent_chunk = Chunk(
                    chunk_id=f"parent-{short_hash(parent_key)}",
                    text="\n\n".join(chunk.text for chunk in siblings),
                    metadata=parent_metadata,
                )
            expanded.append(
                SearchResult(chunk=parent_chunk, score=child_result.score)
            )
            if len(expanded) >= top_k:
                break
        return expanded

    def _get_parent_section(self, parent_id: str | None) -> Chunk | None:
        if not parent_id:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT text, metadata_json
                FROM parent_sections
                WHERE parent_id = ?
                """,
                (parent_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk(
            chunk_id=parent_id,
            text=str(row[0]),
            metadata=ChunkMetadata(**json.loads(row[1])),
        )
