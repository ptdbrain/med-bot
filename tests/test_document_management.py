from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.chunker import TextChunker
from app.main import create_app
from app.rag.service import RAGService
from app.schemas.document import Document


def _settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        index_path=root / "indexes" / "vectors.json",
        metadata_db_path=root / "indexes" / "metadata.db",
    )


class DocumentManagementTests(unittest.TestCase):
    def test_chunker_assigns_stable_parent_id_per_section(self) -> None:
        document = Document(
            doc_id="stroke",
            title="Đột quỵ",
            source_path="stroke.md",
            source_type="md",
            text=(
                "# FAST\n\n"
                + ("Face Arm Speech Time. " * 20)
                + "\n\n## Phòng ngừa\n\nKiểm soát yếu tố nguy cơ."
            ),
        )

        chunks = TextChunker(chunk_size=100, chunk_overlap=10).chunk(document)

        fast_parent_ids = {
            chunk.metadata.parent_id
            for chunk in chunks
            if chunk.metadata.section_title == "FAST"
        }
        prevention_parent_ids = {
            chunk.metadata.parent_id
            for chunk in chunks
            if chunk.metadata.section_title == "Phòng ngừa"
        }
        self.assertEqual(len(fast_parent_ids), 1)
        self.assertNotIn(None, fast_parent_ids)
        self.assertNotEqual(fast_parent_ids, prevention_parent_ids)

    def test_pipeline_persists_parent_sections_for_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = RAGService(_settings(Path(tmp)))
            service.ingest_file(
                "stroke.md",
                (
                    "# FAST\n\n"
                    + ("Face Arm Speech Time là dấu hiệu đột quỵ. " * 25)
                ).encode("utf-8"),
            )

            parents = service.metadata_store.list_parent_sections()
            child_results = service.hybrid_rag.retrieve("FAST đột quỵ", top_k=2)
            expanded = service.metadata_store.expand_parent_results(
                child_results,
                top_k=1,
            )

        self.assertEqual(len(parents), 1)
        self.assertEqual(parents[0]["section_title"], "FAST")
        self.assertGreater(len(expanded[0].chunk.text), len(child_results[0].chunk.text))

    def test_documents_api_lists_ingested_documents_with_chunk_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(_settings(Path(tmp))))
            ingest = client.post(
                "/ingest/file",
                files={
                    "file": (
                        "stroke.md",
                        b"# FAST\n\nFace Arm Speech Time.",
                        "text/markdown",
                    )
                },
            )

            response = client.get("/documents")

        self.assertEqual(ingest.status_code, 200)
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 1)
        self.assertEqual(payload["documents"][0]["title"], "stroke")
        self.assertEqual(payload["documents"][0]["chunks_count"], 1)

    def test_deleting_document_removes_it_from_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = TestClient(create_app(_settings(root)))
            ingest = client.post(
                "/ingest/file",
                files={
                    "file": (
                        "stroke.md",
                        b"# FAST\n\nFace Arm Speech Time.",
                        "text/markdown",
                    )
                },
            ).json()
            raw_path = root / "raw" / "stroke.md"
            self.assertTrue(raw_path.exists())

            deleted = client.delete(f"/documents/{ingest['doc_id']}")
            retrieval = client.get(
                "/retrieve",
                params={"query": "FAST", "rag_mode": "hybrid", "top_k": 3},
            )
            listing = client.get("/documents")
            raw_file_exists_after_delete = raw_path.exists()

        self.assertEqual(deleted.status_code, 200)
        self.assertTrue(deleted.json()["deleted"])
        self.assertEqual(retrieval.json()["results"], [])
        self.assertEqual(listing.json()["documents"], [])
        self.assertFalse(raw_file_exists_after_delete)

    def test_url_ingest_updates_bm25_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            service = RAGService(_settings(Path(tmp)))

            result = service.ingest_url(
                "https://example.org/stroke",
                b"<html><body><h1>FAST</h1><p>Face Arm Speech Time.</p></body></html>",
            )
            matches = service.bm25_store.search("FAST Speech", top_k=1)

        self.assertEqual(result.source_type, "html")
        self.assertEqual(matches[0].chunk.metadata.doc_id, result.doc_id)
        self.assertEqual(
            matches[0].chunk.metadata.extra["url"],
            "https://example.org/stroke",
        )


if __name__ == "__main__":
    unittest.main()
