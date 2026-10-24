from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

import requests


def _sqlite_count(path: Path, table: str) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(path)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]) if exists else 0


def _sqlite_ids(path: Path, table: str, column: str) -> set[str]:
    if not path.exists():
        return set()
    with closing(sqlite3.connect(path)) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            return set()
        return {str(row[0]) for row in connection.execute(f"SELECT {column} FROM {table}")}


def _qdrant_ids(url: str, collection: str, timeout: float = 5.0) -> set[str]:
    endpoint = f"{url.rstrip('/')}/collections/{quote(collection, safe='')}/points/scroll"
    ids: set[str] = set()
    offset = None
    while True:
        body = {"limit": 256, "with_payload": True}
        if offset is not None:
            body["offset"] = offset
        response = requests.post(endpoint, json=body, timeout=timeout)
        response.raise_for_status()
        result = response.json()["result"]
        points = result.get("points", [])
        for point in points:
            chunk_id = point.get("payload", {}).get("chunk", {}).get("chunk_id")
            if chunk_id is not None:
                ids.add(str(chunk_id))
        offset = result.get("next_page_offset")
        if offset is None:
            return ids


def inspect_index(
    metadata_db: Path,
    dense_db: Path,
    memory_index: Path,
    provider: str,
    *,
    qdrant_url: str = "http://localhost:6333",
    qdrant_collection: str = "medtutor_chunks",
) -> dict:
    errors: list[str] = []
    try:
        metadata_chunks = _sqlite_count(metadata_db, "chunks")
        metadata_documents = _sqlite_count(metadata_db, "documents")
        metadata_ids = _sqlite_ids(metadata_db, "chunks", "chunk_id")
        normalized_provider = provider.strip().lower()
        if normalized_provider in {"sqlite-dense", "dense-local"}:
            vector_ids = _sqlite_ids(dense_db, "dense_vectors", "chunk_id")
        elif normalized_provider in {"memory", "local"}:
            if not memory_index.exists():
                vector_ids = set()
            else:
                vector_ids = {
                    str(item["chunk_id"])
                    for item in json.loads(memory_index.read_text(encoding="utf-8"))
                    if item.get("chunk_id") is not None
                }
        elif normalized_provider == "qdrant":
            vector_ids = _qdrant_ids(qdrant_url, qdrant_collection)
        else:
            errors.append("unsupported_vector_store_provider")
            vector_ids = set()
    except (OSError, sqlite3.Error, json.JSONDecodeError, KeyError, ValueError, requests.RequestException) as exc:
        return {
            "passed": False,
            "provider": provider,
            "metadata_documents": 0,
            "metadata_chunks": 0,
            "vectors": 0,
            "missing_vectors": [],
            "orphan_vectors": [],
            "errors": ["index_probe_failed", f"{type(exc).__name__}: {exc}"],
        }

    vectors = len(vector_ids)
    missing_vectors = sorted(metadata_ids - vector_ids)
    orphan_vectors = sorted(vector_ids - metadata_ids)
    if metadata_documents == 0 or metadata_chunks == 0 or vectors == 0:
        errors.append("empty_index")
    if metadata_chunks != vectors:
        errors.append("vector_count_mismatch")
    if missing_vectors or orphan_vectors or len(metadata_ids) != metadata_chunks:
        errors.append("chunk_id_mismatch")
    return {
        "passed": not errors,
        "provider": provider,
        "metadata_documents": metadata_documents,
        "metadata_chunks": metadata_chunks,
        "vectors": vectors,
        "missing_vectors": missing_vectors[:10],
        "orphan_vectors": orphan_vectors[:10],
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    from app.config import get_settings

    settings = get_settings()
    parser = argparse.ArgumentParser(description="Read-only metadata/vector index consistency check")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)
    report = inspect_index(
        settings.metadata_db_path,
        settings.dense_index_path,
        settings.index_path,
        settings.vector_store_provider,
        qdrant_url=settings.qdrant_url,
        qdrant_collection=settings.qdrant_collection,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Index check: {'PASS' if report['passed'] else 'FAIL'}")
        print(
            f"provider={report['provider']} documents={report['metadata_documents']} "
            f"chunks={report['metadata_chunks']} vectors={report['vectors']}"
        )
        for error in report["errors"]:
            print(f"error: {error}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
