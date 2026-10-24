from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from scripts.check_index import inspect_index


class CheckIndexTests(unittest.TestCase):
    def _make_index(self, root: Path, *, chunks: int, vectors: int) -> tuple[Path, Path, Path]:
        metadata = root / "metadata.db"
        connection = sqlite3.connect(metadata)
        connection.execute("CREATE TABLE documents (doc_id TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT)")
        connection.executemany("INSERT INTO documents VALUES (?)", [(f"doc-{i}",) for i in range(chunks)])
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?)",
            [(f"chunk-{i}", f"doc-{i}") for i in range(chunks)],
        )
        connection.commit()
        connection.close()

        dense = root / "dense.db"
        connection = sqlite3.connect(dense)
        connection.execute("CREATE TABLE dense_vectors (chunk_id TEXT PRIMARY KEY)")
        connection.executemany("INSERT INTO dense_vectors VALUES (?)", [(f"chunk-{i}",) for i in range(vectors)])
        connection.commit()
        connection.close()
        return metadata, dense, root / "vectors.json"

    def test_empty_index_is_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata, dense, vectors = self._make_index(Path(tmp), chunks=0, vectors=0)
            report = inspect_index(metadata, dense, vectors, "sqlite-dense")

        self.assertFalse(report["passed"])
        self.assertIn("empty_index", report["errors"])

    def test_dense_count_must_match_metadata_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata, dense, vectors = self._make_index(Path(tmp), chunks=2, vectors=1)
            report = inspect_index(metadata, dense, vectors, "sqlite-dense")

        self.assertFalse(report["passed"])
        self.assertIn("vector_count_mismatch", report["errors"])

    def test_equal_counts_with_different_chunk_ids_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata, dense, vectors = self._make_index(Path(tmp), chunks=1, vectors=1)
            with closing(sqlite3.connect(dense)) as connection:
                connection.execute("UPDATE dense_vectors SET chunk_id = 'orphan'")
                connection.commit()
            report = inspect_index(metadata, dense, vectors, "sqlite-dense")

        self.assertFalse(report["passed"])
        self.assertIn("chunk_id_mismatch", report["errors"])
        self.assertEqual(report["missing_vectors"], ["chunk-0"])
        self.assertEqual(report["orphan_vectors"], ["orphan"])

    def test_matching_nonempty_dense_index_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metadata, dense, vectors = self._make_index(Path(tmp), chunks=2, vectors=2)
            report = inspect_index(metadata, dense, vectors, "sqlite-dense")

        self.assertTrue(report["passed"])
        self.assertEqual(report["metadata_chunks"], 2)
        self.assertEqual(report["vectors"], 2)

    def test_memory_index_uses_persisted_chunk_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metadata, dense, vectors = self._make_index(root, chunks=1, vectors=0)
            vectors.write_text(json.dumps([{"chunk_id": "chunk-0"}]), encoding="utf-8")
            report = inspect_index(metadata, dense, vectors, "memory")

        self.assertTrue(report["passed"])
        self.assertEqual(report["vectors"], 1)


if __name__ == "__main__":
    unittest.main()
