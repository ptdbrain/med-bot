from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.schemas.document import Chunk, Document
from app.utils.file_utils import ensure_directory


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

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return int(row[0])
