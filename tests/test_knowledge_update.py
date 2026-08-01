from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.jobs.crawler import validate_public_url
from app.jobs.incremental_update import IncrementalUrlUpdater
from app.jobs.scheduler import run_batch
from app.rag.service import RAGService


class FakeCrawler:
    def __init__(self, payloads: list[bytes]) -> None:
        self.payloads = payloads
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        return self.payloads.pop(0)


def _service(root: Path) -> RAGService:
    return RAGService(
        Settings(
            data_dir=root,
            index_path=root / "indexes" / "vectors.json",
            metadata_db_path=root / "indexes" / "metadata.db",
        )
    )


class KnowledgeUpdateTests(unittest.TestCase):
    def test_public_url_validation_rejects_local_and_non_http_sources(self) -> None:
        for url in (
            "http://127.0.0.1/private",
            "http://localhost/private",
            "http://10.0.0.5/private",
            "file:///etc/passwd",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "public HTTP"):
                validate_public_url(url)

        self.assertEqual(
            validate_public_url("https://example.org/guideline"),
            "https://example.org/guideline",
        )

    def test_unchanged_url_content_is_not_reingested(self) -> None:
        content = b"<html><body><h1>FAST</h1><p>Face Arm Speech Time.</p></body></html>"
        crawler = FakeCrawler([content, content])
        with tempfile.TemporaryDirectory() as tmp:
            service = _service(Path(tmp))
            updater = IncrementalUrlUpdater(service, crawler)

            first = updater.update("https://example.org/stroke")
            second = updater.update("https://example.org/stroke")

            documents = service.list_documents()

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)
        self.assertEqual(first.doc_id, second.doc_id)
        self.assertEqual(len(documents), 1)

    def test_changed_url_replaces_the_previous_document(self) -> None:
        crawler = FakeCrawler(
            [
                b"<html><body><p>FAST gom Face va Arm.</p></body></html>",
                b"<html><body><p>FAST gom Face, Arm, Speech va Time.</p></body></html>",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            service = _service(Path(tmp))
            updater = IncrementalUrlUpdater(service, crawler)

            first = updater.update("https://example.org/stroke")
            second = updater.update("https://example.org/stroke")
            documents = service.list_documents()
            results = service.retrieve("Speech Time", rag_mode="hybrid", top_k=3)

        self.assertTrue(second.changed)
        self.assertNotEqual(first.doc_id, second.doc_id)
        self.assertEqual([document["doc_id"] for document in documents], [second.doc_id])
        self.assertEqual(results[0].chunk.metadata.doc_id, second.doc_id)

    def test_manual_batch_runs_each_url_once(self) -> None:
        crawler = FakeCrawler(
            [
                b"<html><body><p>FAST stroke.</p></body></html>",
                b"<html><body><p>Hemoglobin blood.</p></body></html>",
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            updater = IncrementalUrlUpdater(_service(Path(tmp)), crawler)

            results = run_batch(
                [
                    "https://example.org/stroke",
                    "https://example.org/blood",
                ],
                updater,
            )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.changed for result in results))


if __name__ == "__main__":
    unittest.main()
