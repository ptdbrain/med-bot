from dataclasses import replace

import pytest

from app.config import Settings
from app.rag.service import RAGService


def settings_for(root):
    return replace(Settings(), data_dir=root, index_path=root / "index.json", metadata_db_path=root / "metadata.db")


def test_failed_replacement_preserves_old_raw_and_retrieval(tmp_path):
    service = RAGService(settings_for(tmp_path))
    old = service.ingest_file("notes.md", b"# Enzymes\n\nEnzymes lower activation energy.")
    old_path = service.metadata_store.get_document(old.doc_id)["source_path"]
    with pytest.raises(ValueError):
        service.ingest_file("notes.md", b"   ", replace_doc_id=old.doc_id)
    from pathlib import Path

    assert "Enzymes" in Path(old_path).read_text()
    assert service.retrieve("Enzymes")


def test_published_revision_replaces_visible_points_and_can_rollback(tmp_path):
    service = RAGService(settings_for(tmp_path))
    old = service.ingest_file("notes.md", b"# Enzymes\n\nEnzymes lower activation energy.")
    first_ids = {c.chunk_id for c in service.vector_store.chunks}
    new = service.ingest_file(
        "notes.md", b"# Enzymes\n\nEnzymes have substrate specificity.", replace_doc_id=old.doc_id
    )
    assert new.doc_id == old.doc_id
    assert first_ids.isdisjoint({c.chunk_id for c in service.vector_store.chunks})
    service.pipeline.publisher.rollback(old.doc_id, old.revision_id)
    assert {c.chunk_id for c in service.vector_store.chunks} == first_ids


def test_embedding_failure_never_publishes_partial_revision(tmp_path, monkeypatch):
    service = RAGService(settings_for(tmp_path))
    old = service.ingest_file("a.md", b"Old valid evidence.")

    def fail(chunks):
        raise RuntimeError("provider down")

    monkeypatch.setattr(service.vector_store, "add_chunks", fail)
    with pytest.raises(RuntimeError):
        service.ingest_file("a.md", b"New evidence should stay invisible.", replace_doc_id=old.doc_id)
    assert "Old valid" in service.metadata_store.list_chunks()[0].text


def test_model_same_dimension_still_requires_reindex():
    from app.retrieval.index_manifest import IndexManifest, require_compatible

    with pytest.raises(ValueError, match="embedding"):
        require_compatible(IndexManifest("2", "a", "c", "e", 1024), IndexManifest("2", "b", "c", "e", 1024))
