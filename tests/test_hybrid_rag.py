from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.rag.rag_fusion import reciprocal_rank_fusion
from app.rag.service import RAGService
from app.retrieval.bm25_store import BM25Store
from app.schemas.document import Chunk, ChunkMetadata, SearchResult


def _chunk(chunk_id: str, text: str, title: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id=title,
            source_path=f"data/sample/{title}.md",
            source_type="md",
            title=title,
            created_at="2026-06-06T00:00:00Z",
            hash=chunk_id,
        ),
    )


class HybridRagTest(unittest.TestCase):
    def test_bm25_ranks_specific_term_above_generic_text(self) -> None:
        store = BM25Store()
        store.add_chunks(
            [
                _chunk("generic-0", "Tai lieu hoc tap ve cac chi so y te.", "generic"),
                _chunk("lab-0", "Hemoglobin van chuyen oxy trong hong cau.", "blood_tests"),
            ]
        )

        results = store.search("hemoglobin oxy", top_k=2)

        self.assertEqual(results[0].chunk.chunk_id, "lab-0")
        self.assertGreater(results[0].score, 0)

    def test_bm25_downweights_repeated_common_terms(self) -> None:
        store = BM25Store()
        store.add_chunks(
            [
                _chunk("generic-0", " ".join(["medical"] * 20), "generic"),
                _chunk("specific-0", "hemoglobin oxygen transport", "blood_tests"),
                _chunk("other-0", "medical education reference", "other"),
            ]
        )

        results = store.search("hemoglobin medical", top_k=3)

        self.assertEqual(results[0].chunk.chunk_id, "specific-0")

    def test_rrf_merges_duplicate_chunks_and_returns_fusion_score(self) -> None:
        stroke = _chunk("stroke-0", "FAST dot quy", "stroke")
        lab = _chunk("lab-0", "Hemoglobin", "blood_tests")

        results = reciprocal_rank_fusion(
            [
                [SearchResult(stroke, 0.9), SearchResult(lab, 0.5)],
                [SearchResult(lab, 5.0), SearchResult(stroke, 2.0)],
            ],
            k=60,
        )

        self.assertEqual({result.chunk.chunk_id for result in results}, {"stroke-0", "lab-0"})
        self.assertTrue(all(0 < result.score < 1 for result in results))

    def test_weighted_rrf_can_prioritize_dense_semantic_ranking(self) -> None:
        semantic = _chunk("semantic-0", "Enzyme catalysis", "biochemistry")
        lexical = _chunk("lexical-0", "Metabolism glossary", "glossary")

        results = reciprocal_rank_fusion(
            [
                [
                    SearchResult(semantic, 0.82, {"semantic_score": 0.82}),
                    SearchResult(lexical, 0.70, {"semantic_score": 0.70}),
                ],
                [
                    SearchResult(lexical, 8.0),
                    SearchResult(semantic, 2.0),
                ],
            ],
            k=10,
            weights=[2.0, 1.0],
            source_names=["dense", "sparse"],
        )

        self.assertEqual(results[0].chunk.chunk_id, "semantic-0")
        self.assertEqual(results[0].signals["dense_rank"], 1.0)
        self.assertEqual(results[0].signals["sparse_rank"], 2.0)
        self.assertEqual(results[0].signals["semantic_score"], 0.82)

    def test_rrf_equal_scores_preserve_configured_source_priority(self) -> None:
        dense = _chunk("z-dense", "Semantic result", "dense")
        sparse = _chunk("a-sparse", "Keyword result", "sparse")

        results = reciprocal_rank_fusion(
            [
                [SearchResult(dense, 0.75)],
                [SearchResult(sparse, 8.0)],
            ],
            source_names=["dense", "sparse"],
        )

        self.assertEqual(results[0].chunk.chunk_id, "z-dense")

    def test_service_hybrid_mode_returns_grounded_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                data_dir=root,
                index_path=root / "indexes" / "vector_store.json",
                metadata_db_path=root / "indexes" / "metadata.db",
            )
            service = RAGService(settings)
            service.ingest_file(
                "blood_tests.md",
                (
                    "# Hemoglobin\n\n" "Hemoglobin phản ánh khả năng vận chuyển oxy của hồng cầu trong xét nghiệm máu."
                ).encode(),
            )

            result = service.chat("Hemoglobin có vai trò gì?", rag_mode="hybrid", top_k=3)

        self.assertEqual(result.mode_used, "hybrid")
        self.assertTrue(result.sources)
        self.assertIn("blood_tests.md", result.sources[0]["source_path"])
        self.assertIn("Thông tin chỉ phục vụ học tập/tham khảo", result.answer)

    def test_service_rebuilds_bm25_from_persisted_vector_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                data_dir=root,
                index_path=root / "indexes" / "vector_store.json",
                metadata_db_path=root / "indexes" / "metadata.db",
            )
            writer = RAGService(settings)
            writer.ingest_file("stroke.md", b"# FAST\n\nFAST gom face arm speech time.")

            restarted = RAGService(settings)
            result = restarted.chat("FAST gồm gì?", rag_mode="hybrid", top_k=2)

        self.assertTrue(result.sources)
        self.assertEqual(result.mode_used, "hybrid")


if __name__ == "__main__":
    unittest.main()
