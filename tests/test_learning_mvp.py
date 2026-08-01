from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip(
    "learning_mvp.core",
    reason="learning_mvp was removed from this workspace",
)

from learning_mvp.core import (
    answer_from_chunks,
    build_index,
    chunk_document,
    load_chunks,
    read_document,
    save_chunks,
    search_chunks,
    search_chunks_tfidf,
)


class LearningMvpTest(unittest.TestCase):
    def test_read_document_returns_text_and_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "note.md"
            path.write_text("# Stroke\n\nFAST helps recognize stroke.", encoding="utf-8")

            document = read_document(path)

        self.assertEqual(document["title"], "note")
        self.assertEqual(document["source_path"], str(path))
        self.assertIn("FAST helps", document["text"])

    def test_chunk_document_creates_small_chunks_with_metadata(self) -> None:
        document = {
            "title": "stroke",
            "source_path": "data/sample/stroke.md",
            "text": "FAST helps recognize stroke.\n\nFace arm speech time.\n\nCall emergency early.",
        }

        chunks = chunk_document(document, max_chars=40)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["source_path"], "data/sample/stroke.md")
        self.assertEqual(chunks[0]["chunk_index"], 0)
        self.assertIn("chunk_id", chunks[0])

    def test_save_and_load_chunks_roundtrip_json(self) -> None:
        chunks = [
            {
                "chunk_id": "stroke-0",
                "chunk_index": 0,
                "text": "FAST stroke signs",
                "source_path": "data/sample/stroke.md",
                "title": "stroke",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.json"
            save_chunks(chunks, path)

            loaded = load_chunks(path)

        self.assertEqual(loaded, chunks)

    def test_search_chunks_returns_best_keyword_match_first(self) -> None:
        chunks = [
            {
                "chunk_id": "lab-0",
                "chunk_index": 0,
                "text": "Hemoglobin is a blood test value.",
                "source_path": "data/sample/blood_tests.md",
                "title": "blood_tests",
            },
            {
                "chunk_id": "stroke-0",
                "chunk_index": 0,
                "text": "FAST includes face arm speech time for stroke.",
                "source_path": "data/sample/stroke.md",
                "title": "stroke",
            },
        ]

        results = search_chunks("FAST stroke signs", chunks, top_k=1)

        self.assertEqual(results[0]["chunk_id"], "stroke-0")
        self.assertGreater(results[0]["score"], 0)

    def test_search_chunks_ignores_common_vietnamese_question_words(self) -> None:
        chunks = [
            {
                "chunk_id": "stroke-0",
                "chunk_index": 0,
                "text": "Tài liệu này nói về FAST là cách ghi nhớ dấu hiệu đột quỵ.",
                "source_path": "data/sample/stroke.md",
                "title": "stroke",
            },
            {
                "chunk_id": "lab-0",
                "chunk_index": 0,
                "text": "Hemoglobin phản ánh khả năng vận chuyển oxy của hồng cầu.",
                "source_path": "data/sample/blood_tests.md",
                "title": "blood_tests",
            },
        ]

        results = search_chunks("Tài liệu nói gì về hemoglobin?", chunks, top_k=1)

        self.assertEqual(results[0]["chunk_id"], "lab-0")

    def test_tfidf_search_prefers_specific_rare_terms(self) -> None:
        chunks = [
            {
                "chunk_id": "generic-0",
                "chunk_index": 0,
                "text": "Xét nghiệm là tài liệu học tập về nhiều chỉ số y tế chung.",
                "source_path": "data/sample/generic.md",
                "title": "generic",
            },
            {
                "chunk_id": "hemoglobin-0",
                "chunk_index": 0,
                "text": "Xét nghiệm hemoglobin phản ánh khả năng vận chuyển oxy của hồng cầu.",
                "source_path": "data/sample/blood_tests.md",
                "title": "blood_tests",
            },
        ]

        results = search_chunks_tfidf("hemoglobin xét nghiệm", chunks, top_k=2)

        self.assertEqual(results[0]["chunk_id"], "hemoglobin-0")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_answer_from_chunks_reports_sources_or_missing_context(self) -> None:
        chunks = [
            {
                "chunk_id": "stroke-0",
                "chunk_index": 0,
                "text": "FAST includes face arm speech time.",
                "source_path": "data/sample/stroke.md",
                "title": "stroke",
                "score": 2,
            }
        ]

        answer = answer_from_chunks("FAST là gì?", chunks)
        empty_answer = answer_from_chunks("Không có gì?", [])

        self.assertIn("Nguồn:", answer)
        self.assertIn("data/sample/stroke.md", answer)
        self.assertIn("chưa tìm thấy", empty_answer)

    def test_numbered_scripts_run_directly_from_repo_root(self) -> None:
        result = subprocess.run(
            [sys.executable, "learning_mvp/01_ingest.py"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DOCUMENT", result.stdout)

    def test_chat_script_supports_tfidf_search_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            chunks_path = Path(tmp) / "chunks.json"
            save_chunks(
                [
                    {
                        "chunk_id": "lab-0",
                        "chunk_index": 0,
                        "text": "Xét nghiệm hemoglobin phản ánh vận chuyển oxy.",
                        "source_path": "data/sample/blood_tests.md",
                        "title": "blood_tests",
                    }
                ],
                chunks_path,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "learning_mvp/05_chat.py",
                    "hemoglobin là gì?",
                    "--chunks",
                    str(chunks_path),
                    "--search-mode",
                    "tfidf",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Nguồn:", result.stdout)
        self.assertIn("lab-0", result.stdout)

    def test_build_index_reads_multiple_documents_and_keeps_unique_chunk_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stroke.md").write_text("FAST stroke signs.\n\nCall emergency early.", encoding="utf-8")
            (root / "lab.txt").write_text("Hemoglobin is a blood test value.", encoding="utf-8")

            chunks = build_index(root, max_chars=60)

        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        source_paths = {Path(chunk["source_path"]).name for chunk in chunks}
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(source_paths, {"stroke.md", "lab.txt"})


if __name__ == "__main__":
    unittest.main()
