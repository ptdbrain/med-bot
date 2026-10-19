from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings


def _reconcile_memory_store(
    store_path: Path,
    sqlite_chunk_ids: set[str],
) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not store_path.is_file():
        if sqlite_chunk_ids:
            errors.append(
                f"Vector store index file missing at {store_path} ({len(sqlite_chunk_ids)} chunks expected)"
            )
        return 0, errors
    try:
        data = json.loads(store_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            errors.append(f"Vector store index file at {store_path} is invalid (expected JSON list)")
            return 0, errors
        vectors_count = len(data)
        vector_chunk_ids = {
            item["chunk_id"] for item in data if isinstance(item, dict) and "chunk_id" in item
        }
        missing_in_vs = sqlite_chunk_ids - vector_chunk_ids
        extra_in_vs = vector_chunk_ids - sqlite_chunk_ids
        if missing_in_vs:
            errors.append(f"Vector store missing {len(missing_in_vs)} vectors present in metadata")
        if extra_in_vs:
            errors.append(f"Vector store contains {len(extra_in_vs)} extra vectors not in metadata")
        if vectors_count != len(sqlite_chunk_ids):
            errors.append(
                f"Vector store vector count ({vectors_count}) does not match chunk count ({len(sqlite_chunk_ids)})"
            )
        return vectors_count, errors
    except (OSError, json.JSONDecodeError, TypeError, KeyError) as exc:
        errors.append(f"Failed to read vector store at {store_path}: {exc}")
        return 0, errors


def _reconcile_dense_store(
    dense_path: Path,
    sqlite_chunk_ids: set[str],
    settings: Settings | None = None,
) -> tuple[int, list[str]]:
    errors: list[str] = []
    if not dense_path.is_file():
        if sqlite_chunk_ids:
            errors.append(
                f"Dense vector database missing at {dense_path} ({len(sqlite_chunk_ids)} chunks expected)"
            )
        return 0, errors
    try:
        uri = f"file:{dense_path.resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            dense_tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if "dense_vectors" not in dense_tables:
                errors.append(f"Dense vector DB at {dense_path} is missing 'dense_vectors' table")
                return 0, errors
            vectors_count = conn.execute("SELECT COUNT(*) FROM dense_vectors").fetchone()[0]
            dense_chunk_ids = {str(row[0]) for row in conn.execute("SELECT chunk_id FROM dense_vectors")}
            missing_in_vs = sqlite_chunk_ids - dense_chunk_ids
            extra_in_vs = dense_chunk_ids - sqlite_chunk_ids
            if missing_in_vs:
                errors.append(f"Dense vector store missing {len(missing_in_vs)} vectors present in metadata")
            if extra_in_vs:
                errors.append(f"Dense vector store contains {len(extra_in_vs)} extra vectors not in metadata")
            if vectors_count != len(sqlite_chunk_ids):
                errors.append(
                    f"Dense vector count ({vectors_count}) does not match chunk count ({len(sqlite_chunk_ids)})"
                )
            if settings and "dense_metadata" in dense_tables:
                dim_row = conn.execute("SELECT value FROM dense_metadata WHERE key='vector_size'").fetchone()
                if dim_row and int(dim_row[0]) != settings.embedding_dimension:
                    errors.append(
                        f"Dense vector dimension mismatch: stored={dim_row[0]} vs configured={settings.embedding_dimension}"
                    )
            return vectors_count, errors
        finally:
            conn.close()
    except (sqlite3.DatabaseError, OSError) as exc:
        errors.append(f"Failed to inspect dense vector store at {dense_path}: {exc}")
        return 0, errors


def _reconcile_qdrant_store(
    settings: Settings,
    sqlite_chunk_ids: set[str],
) -> tuple[int | None, list[str]]:
    errors: list[str] = []
    import requests

    try:
        from app.retrieval.index_manifest import embedding_fingerprint

        collection = settings.qdrant_collection + "_" + embedding_fingerprint(settings)[:16]
        url = f"{settings.qdrant_url.rstrip('/')}/collections/{collection}"
        resp = requests.get(url, timeout=3)
        if resp.status_code == 404:
            if sqlite_chunk_ids:
                errors.append(
                    f"Qdrant collection '{collection}' not found ({len(sqlite_chunk_ids)} chunks expected)"
                )
            return 0, errors
        if resp.status_code != 200:
            errors.append(f"Qdrant collection check failed with HTTP {resp.status_code}")
            return None, errors
        payload = resp.json().get("result", {})
        vectors_count = payload.get("points_count", payload.get("vectors_count", 0))
        if vectors_count != len(sqlite_chunk_ids):
            errors.append(
                f"Qdrant points count ({vectors_count}) does not match chunk count ({len(sqlite_chunk_ids)})"
            )
        return vectors_count, errors
    except (requests.RequestException, ValueError, TypeError) as exc:
        if sqlite_chunk_ids:
            errors.append(f"Failed to connect to Qdrant at {settings.qdrant_url}: {exc}")
        return None, errors


def _reconcile_vector_store_instance(
    vector_store: Any,
    sqlite_chunk_ids: set[str],
) -> tuple[int, list[str]]:
    errors: list[str] = []
    vector_chunk_ids: set[str] | None = None
    if hasattr(vector_store, "existing_chunk_ids") and callable(vector_store.existing_chunk_ids):
        vector_chunk_ids = set(vector_store.existing_chunk_ids())
        vectors_count = len(vector_chunk_ids)
    elif hasattr(vector_store, "chunks"):
        vector_chunk_ids = {c.chunk_id for c in vector_store.chunks}
        vectors_count = len(vector_chunk_ids)
    elif hasattr(vector_store, "count"):
        vectors_count = vector_store.count
    else:
        vectors_count = 0
        errors.append("Vector store object does not expose count or chunks")

    if vector_chunk_ids is not None:
        missing_in_vs = sqlite_chunk_ids - vector_chunk_ids
        extra_in_vs = vector_chunk_ids - sqlite_chunk_ids
        if missing_in_vs:
            errors.append(f"Vector store missing {len(missing_in_vs)} vectors present in metadata")
        if extra_in_vs:
            errors.append(f"Vector store contains {len(extra_in_vs)} extra vectors not in metadata")
    if vectors_count != len(sqlite_chunk_ids):
        errors.append(
            f"Vector store vector count ({vectors_count}) does not match chunk count ({len(sqlite_chunk_ids)})"
        )
    return vectors_count, errors


def reconcile_vectors(
    sqlite_chunk_ids: set[str],
    db_path: Path,
    settings: Settings | None = None,
    vector_store: Any | None = None,
) -> tuple[int | None, list[str]]:
    if vector_store is not None:
        return _reconcile_vector_store_instance(vector_store, sqlite_chunk_ids)

    if settings is None:
        try:
            settings = get_settings()
        except (ValueError, OSError):
            settings = None

    peer_memory = db_path.parent / "vector_store.json"
    peer_dense = db_path.parent / "dense_vectors.db"

    provider = settings.vector_store_provider.strip().lower() if settings else "memory"

    try:
        is_same_db = settings is not None and settings.metadata_db_path.resolve() == db_path.resolve()
    except (OSError, RuntimeError):
        is_same_db = False

    if provider in {"memory", "local"}:
        index_path = (
            settings.index_path
            if (is_same_db and settings is not None)
            else (peer_memory if peer_memory.is_file() else (settings.index_path if settings else peer_memory))
        )
        return _reconcile_memory_store(index_path, sqlite_chunk_ids)

    if provider in {"sqlite-dense", "dense-local"}:
        dense_path = (
            settings.dense_index_path
            if (is_same_db and settings is not None)
            else (peer_dense if peer_dense.is_file() else (settings.dense_index_path if settings else peer_dense))
        )
        return _reconcile_dense_store(dense_path, sqlite_chunk_ids, settings=settings)

    if provider == "qdrant":
        if settings:
            return _reconcile_qdrant_store(settings, sqlite_chunk_ids)
        return None, ["Cannot reconcile Qdrant vector store without settings"]

    if peer_memory.is_file():
        return _reconcile_memory_store(peer_memory, sqlite_chunk_ids)
    if peer_dense.is_file():
        return _reconcile_dense_store(peer_dense, sqlite_chunk_ids, settings=settings)

    if settings:
        return _reconcile_memory_store(settings.index_path, sqlite_chunk_ids)
    return 0, ["Vector store file not found"] if sqlite_chunk_ids else (0, [])


def inspect_index(
    db_path: Path,
    *,
    read_only: bool = True,
    settings: Settings | None = None,
    vector_store: Any | None = None,
    check_vectors: bool = True,
) -> dict:
    if not db_path.is_file():
        return {
            "status": "missing",
            "metadata_db": str(db_path),
            "documents": 0,
            "chunks": 0,
            "active_revisions": 0,
            "vectors": 0,
            "details": f"Metadata database file not found at {db_path}",
        }

    uri = f"file:{db_path.resolve().as_posix()}?mode=ro" if read_only else str(db_path)

    try:
        connection = sqlite3.connect(uri, uri=read_only)
    except sqlite3.DatabaseError as exc:
        return {
            "status": "invalid",
            "metadata_db": str(db_path),
            "documents": 0,
            "chunks": 0,
            "active_revisions": 0,
            "vectors": 0,
            "details": f"Failed to connect to SQLite database: {exc}",
        }

    try:
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        except sqlite3.DatabaseError as exc:
            return {
                "status": "invalid",
                "metadata_db": str(db_path),
                "documents": 0,
                "chunks": 0,
                "active_revisions": 0,
                "vectors": 0,
                "details": f"Database corrupted: {exc}",
            }

        required_tables = {"documents", "chunks"}
        if not required_tables.issubset(tables):
            return {
                "status": "invalid",
                "metadata_db": str(db_path),
                "missing_tables": sorted(required_tables - tables),
                "documents": 0,
                "chunks": 0,
                "active_revisions": 0,
                "vectors": 0,
                "details": f"Missing required tables: {sorted(required_tables - tables)}",
            }

        doc_cols = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        chunk_cols = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        missing_doc_cols = {"doc_id", "title", "source_path"} - doc_cols
        missing_chunk_cols = {"chunk_id", "doc_id", "text"} - chunk_cols
        if missing_doc_cols or missing_chunk_cols:
            return {
                "status": "invalid",
                "metadata_db": str(db_path),
                "documents": 0,
                "chunks": 0,
                "active_revisions": 0,
                "vectors": 0,
                "details": (
                    f"Invalid table schema (missing columns: "
                    f"docs={sorted(missing_doc_cols)}, chunks={sorted(missing_chunk_cols)})"
                ),
            }

        docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        has_active_revs = "active_revisions" in tables
        has_index_revs = "index_revisions" in tables
        active = (
            connection.execute("SELECT COUNT(*) FROM active_revisions").fetchone()[0]
            if has_active_revs
            else None
        )

        chunk_rows = connection.execute("SELECT chunk_id, doc_id, text, metadata_json FROM chunks").fetchall()
        sqlite_chunk_ids = {str(row[0]) for row in chunk_rows}

        vectors_count: int | None = None
        vector_errors: list[str] = []
        if check_vectors:
            vectors_count, vector_errors = reconcile_vectors(
                sqlite_chunk_ids=sqlite_chunk_ids,
                db_path=db_path,
                settings=settings,
                vector_store=vector_store,
            )

        # Check for empty index
        if docs == 0 and chunks == 0:
            if (vectors_count or 0) > 0 or vector_errors:
                return {
                    "status": "partial",
                    "metadata_db": str(db_path),
                    "documents": 0,
                    "chunks": 0,
                    "active_revisions": active if active is not None else 0,
                    "vectors": vectors_count if vectors_count is not None else 0,
                    "details": (
                        f"Metadata is empty (0 docs, 0 chunks) but vector store contains "
                        f"{vectors_count} orphaned vectors"
                    ),
                    "reconciliation_errors": vector_errors or ["Orphan vectors in vector store"],
                }
            return {
                "status": "empty",
                "metadata_db": str(db_path),
                "documents": 0,
                "chunks": 0,
                "active_revisions": active if active is not None else 0,
                "vectors": vectors_count if vectors_count is not None else 0,
                "details": "Index is empty (0 documents, 0 chunks)",
            }

        reconciliation_errors: list[str] = []

        # 1. Document vs Chunk count checks
        if docs > 0 and chunks == 0:
            reconciliation_errors.append(f"Documents exist ({docs}) but chunks count is 0")
        if docs == 0 and chunks > 0:
            reconciliation_errors.append(f"Chunks exist ({chunks}) but documents count is 0")

        # 2. Orphan chunks check (chunk referencing nonexistent doc)
        orphan_chunks = connection.execute(
            "SELECT COUNT(*) FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
        ).fetchone()[0]
        if orphan_chunks > 0:
            reconciliation_errors.append(f"Found {orphan_chunks} orphan chunks referencing nonexistent documents")

        # 3. Documents without chunks
        empty_docs = connection.execute(
            "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = d.doc_id)"
        ).fetchone()[0]
        if empty_docs > 0:
            reconciliation_errors.append(f"Found {empty_docs} documents with zero chunks")

        # 4. Chunk text and metadata integrity
        empty_text_chunks = 0
        malformed_metadata = 0
        for row in chunk_rows:
            _cid, _, text, raw_meta = row[0], row[1], row[2], row[3]
            if not text or not str(text).strip():
                empty_text_chunks += 1
            if raw_meta:
                try:
                    meta = json.loads(raw_meta)
                    if not isinstance(meta, dict):
                        malformed_metadata += 1
                except (json.JSONDecodeError, TypeError):
                    malformed_metadata += 1
        if empty_text_chunks > 0:
            reconciliation_errors.append(f"Found {empty_text_chunks} chunks with empty or whitespace-only text")
        if malformed_metadata > 0:
            reconciliation_errors.append(f"Found {malformed_metadata} chunks with malformed metadata_json")

        # 5. Active revisions reconciliation
        if has_active_revs:
            if docs > 0:
                if active == 0:
                    reconciliation_errors.append("Documents exist but no active revisions are published")
                elif active != docs:
                    reconciliation_errors.append(
                        f"Active revisions count ({active}) does not match documents count ({docs})"
                    )

                orphan_revs = connection.execute(
                    "SELECT COUNT(*) FROM active_revisions WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
                ).fetchone()[0]
                if orphan_revs > 0:
                    reconciliation_errors.append(
                        f"Found {orphan_revs} active revisions referencing nonexistent documents"
                    )

                docs_missing_revs = connection.execute(
                    "SELECT COUNT(*) FROM documents WHERE doc_id NOT IN (SELECT doc_id FROM active_revisions)"
                ).fetchone()[0]
                if docs_missing_revs > 0:
                    reconciliation_errors.append(
                        f"Found {docs_missing_revs} documents missing from active_revisions"
                    )

            if has_index_revs and active and active > 0:
                unready = connection.execute(
                    "SELECT COUNT(*) FROM active_revisions ar "
                    "JOIN index_revisions ir ON ar.revision_id = ir.revision_id "
                    "WHERE ir.status != 'published'"
                ).fetchone()[0]
                if unready > 0:
                    reconciliation_errors.append(
                        f"Found {unready} active revisions that are not marked as 'published' in index_revisions"
                    )

                missing_revisions = connection.execute(
                    "SELECT COUNT(*) FROM active_revisions "
                    "WHERE revision_id NOT IN (SELECT revision_id FROM index_revisions)"
                ).fetchone()[0]
                if missing_revisions > 0:
                    reconciliation_errors.append(
                        f"Found {missing_revisions} active revisions missing from index_revisions"
                    )

        # 6. Vector store reconciliation errors
        reconciliation_errors.extend(vector_errors)

        if reconciliation_errors:
            return {
                "status": "partial",
                "metadata_db": str(db_path),
                "documents": docs,
                "chunks": chunks,
                "active_revisions": active,
                "vectors": vectors_count,
                "details": "; ".join(reconciliation_errors),
                "reconciliation_errors": reconciliation_errors,
            }

        return {
            "status": "ok",
            "metadata_db": str(db_path),
            "documents": docs,
            "chunks": chunks,
            "active_revisions": active,
            "vectors": vectors_count,
            "details": "Index and vector store reconciled successfully",
        }
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect index status and reconcile vector store.")
    parser.add_argument(
        "--read-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open SQLite in read-only mode",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        default=False,
        help="Allow empty index for fresh installations (exit code 0 instead of 1 when empty)",
    )
    parser.add_argument("--metadata-db", type=Path, default=None, help="Custom path to metadata SQLite database")
    parser.add_argument(
        "--skip-vector-check",
        action="store_true",
        default=False,
        help="Skip vector store reconciliation check",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    db_path = args.metadata_db or settings.metadata_db_path

    result = inspect_index(
        db_path,
        read_only=args.read_only,
        settings=settings,
        check_vectors=not args.skip_vector_check,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for key, value in result.items():
            if isinstance(value, (dict, list)):
                print(f"{key}: {json.dumps(value, ensure_ascii=False)}")
            else:
                print(f"{key}: {value}")

    if result["status"] == "ok":
        return 0
    if result["status"] == "empty" and args.allow_empty:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
