from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from app.config import Settings
from scripts.check_index import inspect_index, main


def _create_metadata_schema(db_path: Path, with_revisions: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            source_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    if with_revisions:
        conn.execute(
            """
            CREATE TABLE active_revisions (
                doc_id TEXT PRIMARY KEY,
                revision_id TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE index_revisions (
                revision_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                previous TEXT
            )
            """
        )
    conn.commit()
    return conn


def test_missing_database_returns_missing_status(tmp_path: Path):
    db_path = tmp_path / "nonexistent.db"
    result = inspect_index(db_path)

    assert result["status"] == "missing"
    assert result["documents"] == 0
    assert result["chunks"] == 0
    assert "not found" in result["details"]

    # Exit code must be 1 even with --allow-empty
    assert main(["--metadata-db", str(db_path)]) == 1
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 1


def test_corrupt_database_returns_invalid_status(tmp_path: Path):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"NOT A SQLITE FILE")

    result = inspect_index(db_path)
    assert result["status"] == "invalid"
    assert main(["--metadata-db", str(db_path)]) == 1
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 1


def test_missing_required_tables_returns_invalid_status(tmp_path: Path):
    db_path = tmp_path / "incomplete.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE other_table (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    result = inspect_index(db_path)
    assert result["status"] == "invalid"
    assert "documents" in result.get("missing_tables", [])
    assert "chunks" in result.get("missing_tables", [])
    assert main(["--metadata-db", str(db_path)]) == 1
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 1


def test_missing_required_columns_returns_invalid_status(tmp_path: Path):
    db_path = tmp_path / "bad_schema.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE documents (wrong_col TEXT)")
    conn.execute("CREATE TABLE chunks (wrong_col TEXT)")
    conn.commit()
    conn.close()

    result = inspect_index(db_path)
    assert result["status"] == "invalid"
    assert "missing columns" in result["details"]
    assert main(["--metadata-db", str(db_path)]) == 1


def test_empty_database_returns_empty_status_and_enforces_exit_code(tmp_path: Path):
    db_path = tmp_path / "empty.db"
    conn = _create_metadata_schema(db_path)
    conn.close()

    result = inspect_index(db_path)
    assert result["status"] == "empty"
    assert result["documents"] == 0
    assert result["chunks"] == 0
    assert result["vectors"] == 0

    # Normal quality gate without --allow-empty MUST FAIL (exit code 1)
    assert main(["--metadata-db", str(db_path)]) == 1

    # Fresh setup WITH --allow-empty SUCCEEDS (exit code 0)
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 0


def test_empty_metadata_with_orphaned_vectors_is_partial(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.close()

    vector_path = tmp_path / "vector_store.json"
    vector_path.write_text(
        json.dumps([{"chunk_id": "orphan-vector-1", "text": "hello"}]),
        encoding="utf-8",
    )

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert result["vectors"] == 1
    assert "orphaned vectors" in result["details"].lower()
    assert main(["--metadata-db", str(db_path)]) == 1
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 1


def test_partial_when_documents_exist_but_no_chunks(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute(
        "INSERT INTO documents VALUES ('doc-1', 'Title 1', 'path.pdf', 'pdf', '{}')"
    )
    conn.commit()
    conn.close()

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert any("chunks count is 0" in err for err in result.get("reconciliation_errors", []))
    assert main(["--metadata-db", str(db_path)]) == 1
    assert main(["--metadata-db", str(db_path), "--allow-empty"]) == 1


def test_partial_when_orphaned_chunks_exist(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    # Chunk with doc-999 which does not exist in documents
    conn.execute(
        "INSERT INTO chunks VALUES ('c-1', 'doc-999', 'some text', '{\"title\": \"X\"}')"
    )
    conn.commit()
    conn.close()

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert any("orphan chunks" in err for err in result.get("reconciliation_errors", []))
    assert main(["--metadata-db", str(db_path)]) == 1


def test_partial_when_active_revisions_mismatched(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO documents VALUES ('doc-2', 'Doc 2', 'p2.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-2', 'doc-2', 'text 2', '{}')")
    # Only doc-1 has an active revision, doc-2 is missing
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'published', '{}', NULL)")
    conn.commit()
    conn.close()

    vector_path = tmp_path / "vector_store.json"
    vector_path.write_text(
        json.dumps([{"chunk_id": "c-1"}, {"chunk_id": "c-2"}]),
        encoding="utf-8",
    )

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert any("Active revisions count" in err or "missing from active_revisions" in err for err in result.get("reconciliation_errors", []))


def test_partial_when_active_revision_not_published(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{}')")
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    # Status is staging instead of published
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'staging', '{}', NULL)")
    conn.commit()
    conn.close()

    vector_path = tmp_path / "vector_store.json"
    vector_path.write_text(json.dumps([{"chunk_id": "c-1"}]), encoding="utf-8")

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert any("not marked as 'published'" in err for err in result.get("reconciliation_errors", []))


def test_partial_when_vector_store_file_missing(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{}')")
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'published', '{}', NULL)")
    conn.commit()
    conn.close()

    # Do not create vector_store.json
    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert any("missing" in err for err in result.get("reconciliation_errors", []))


def test_partial_when_vector_store_count_or_ids_mismatch(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-2', 'doc-1', 'text 2', '{}')")
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'published', '{}', NULL)")
    conn.commit()
    conn.close()

    # Vector store only has c-1, missing c-2
    vector_path = tmp_path / "vector_store.json"
    vector_path.write_text(json.dumps([{"chunk_id": "c-1"}]), encoding="utf-8")

    result = inspect_index(db_path)
    assert result["status"] == "partial"
    assert result["vectors"] == 1
    assert result["chunks"] == 2
    assert any("missing" in err and "vectors" in err for err in result.get("reconciliation_errors", []))


def test_ok_when_fully_reconciled_with_memory_store(tmp_path: Path):
    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{\"doc_id\": \"doc-1\"}')")
    conn.execute("INSERT INTO chunks VALUES ('c-2', 'doc-1', 'text 2', '{\"doc_id\": \"doc-1\"}')")
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'published', '{}', NULL)")
    conn.commit()
    conn.close()

    vector_path = tmp_path / "vector_store.json"
    vector_path.write_text(
        json.dumps([{"chunk_id": "c-1"}, {"chunk_id": "c-2"}]),
        encoding="utf-8",
    )

    result = inspect_index(db_path)
    assert result["status"] == "ok"
    assert result["documents"] == 1
    assert result["chunks"] == 2
    assert result["active_revisions"] == 1
    assert result["vectors"] == 2
    assert main(["--metadata-db", str(db_path)]) == 0


def test_ok_when_fully_reconciled_with_dense_store(tmp_path: Path):
    import struct

    db_path = tmp_path / "metadata.db"
    conn = _create_metadata_schema(db_path)
    conn.execute("INSERT INTO documents VALUES ('doc-1', 'Doc 1', 'p1.md', 'md', '{}')")
    conn.execute("INSERT INTO chunks VALUES ('c-1', 'doc-1', 'text 1', '{}')")
    conn.execute("INSERT INTO active_revisions VALUES ('doc-1', 'rev-1')")
    conn.execute("INSERT INTO index_revisions VALUES ('rev-1', 'doc-1', 'published', '{}', NULL)")
    conn.commit()
    conn.close()

    dense_path = tmp_path / "dense_vectors.db"
    dconn = sqlite3.connect(dense_path)
    dconn.execute(
        "CREATE TABLE dense_vectors (chunk_id TEXT PRIMARY KEY, vector BLOB NOT NULL, chunk_json TEXT NOT NULL)"
    )
    dconn.execute("CREATE TABLE dense_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    packed = struct.pack("<4f", 0.1, 0.2, 0.3, 0.4)
    dconn.execute("INSERT INTO dense_vectors VALUES ('c-1', ?, '{}')", (packed,))
    dconn.execute("INSERT INTO dense_metadata VALUES ('vector_size', '4')")
    dconn.commit()
    dconn.close()

    settings = Settings(
        vector_store_provider="sqlite-dense",
        dense_index_path=dense_path,
        metadata_db_path=db_path,
        embedding_dimension=4,
    )
    result = inspect_index(db_path, settings=settings)
    assert result["status"] == "ok"
    assert result["vectors"] == 1
    assert result["chunks"] == 1


def test_main_cli_json_output(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "empty.db"
    conn = _create_metadata_schema(db_path)
    conn.close()

    exit_code = main(["--metadata-db", str(db_path), "--json", "--allow-empty"])
    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "empty"
    assert payload["documents"] == 0
