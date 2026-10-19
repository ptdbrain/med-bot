from __future__ import annotations

import json
from dataclasses import asdict

from app.ingestion.identity import identity_hash
from app.schemas.document import Chunk, Document


class ActiveVectorStore:
    """Staged points remain invisible until SQLite publishes their chunk IDs."""

    def __init__(self, store, metadata):
        self.store, self.metadata = store, metadata

    def add_chunks(self, chunks):
        self.store.add_chunks(chunks)

    def delete_chunks(self, ids):
        self.store.delete_chunks(ids)

    @property
    def chunks(self):
        active = {c.chunk_id for c in self.metadata.list_chunks()}
        return [c for c in getattr(self.store, "chunks", []) if c.chunk_id in active]

    @property
    def count(self):
        ids = getattr(self.store, "existing_chunk_ids", None)
        if callable(ids):
            return len(ids() & {c.chunk_id for c in self.metadata.list_chunks()})
        if hasattr(self.store, "chunks"):
            return len(self.chunks)
        return None

    def search(self, query, top_k=5, min_score=0.0):
        if top_k <= 0:
            return []
        active = {c.chunk_id for c in self.metadata.list_chunks()}
        if not active:
            return []
        size = max(top_k * 4, 40)
        while True:
            candidates = self.store.search(query, top_k=size, min_score=min_score)
            selected = [r for r in candidates if r.chunk.chunk_id in active]
            if len(selected) >= top_k or len(candidates) < size:
                return selected[:top_k]
            if size >= 65536:
                raise RuntimeError("Too many unpublished vectors; run index recovery")
            size *= 2


class IndexPublisher:
    def __init__(self, metadata, vectors):
        self.metadata, self.vectors = metadata, vectors

    def active_revision(self, doc_id):
        with self.metadata._connect() as connection:
            row = connection.execute("SELECT revision_id FROM active_revisions WHERE doc_id=?", (doc_id,)).fetchone()
        return row[0] if row else None

    def stage(self, document, chunks, parents=None, report=None):
        if not chunks or any(not c.text.strip() for c in chunks):
            raise ValueError("Cannot publish empty chunks")
        previous = self.active_revision(document.doc_id)
        revision = (
            "rev-"
            + identity_hash(
                [document.doc_id, document.metadata.get("content_hash", ""), *[c.chunk_id for c in chunks]]
            )[:32]
        )
        for chunk in chunks:
            chunk.chunk_id = "chunk-" + identity_hash([revision, chunk.chunk_id])[:32]
            chunk.metadata.extra["revision_id"] = revision
        document.metadata["revision_id"] = revision
        payload = json.dumps(
            {
                "document": asdict(document),
                "chunks": [c.to_dict() for c in chunks],
                "previous": previous,
                "report": report or {},
            },
            ensure_ascii=False,
        )
        with self.metadata._connect() as connection:
            row = connection.execute("SELECT status FROM index_revisions WHERE revision_id=?", (revision,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO index_revisions VALUES(?,?,?,?,?)",
                    (revision, document.doc_id, "staging", payload, previous),
                )
        try:
            self.vectors.add_chunks(chunks)
        except Exception:
            with self.metadata._connect() as connection:
                connection.execute(
                    "UPDATE index_revisions SET status='failed' WHERE revision_id=? AND status!='published'",
                    (revision,),
                )
            raise
        with self.metadata._connect() as connection:
            connection.execute(
                "UPDATE index_revisions SET status='ready' WHERE revision_id=? AND status!='published'", (revision,)
            )
        return revision

    def publish(self, doc_id, revision_id, expected_previous):
        with self.metadata._connect() as connection:
            row = connection.execute(
                "SELECT payload,status FROM index_revisions WHERE revision_id=? AND doc_id=?", (revision_id, doc_id)
            ).fetchone()
        if not row or row[1] not in {"ready", "published"}:
            raise ValueError("Revision is not ready")
        payload = json.loads(row[0])
        document = Document(**payload["document"])
        chunks = [Chunk.from_dict(c) for c in payload["chunks"]]
        self.metadata.replace_document_chunks(document, chunks, publication=(revision_id, expected_previous))

    def rollback(self, doc_id, revision_id):
        self.publish(doc_id, revision_id, self.active_revision(doc_id))

    def recover(self):
        with self.metadata._connect() as connection:
            rows = connection.execute(
                "SELECT revision_id,doc_id,status FROM index_revisions WHERE status IN ('staging','failed','ready')"
            ).fetchall()
        return {"pending": [{"revision_id": r[0], "doc_id": r[1], "status": r[2]} for r in rows]}
