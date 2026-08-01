from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.agentic_router import RuleBasedAgenticRouter
from app.rag.corrective_rag import CorrectiveRAG
from app.rag.multi_query_rag import MultiQueryRAG
from app.rag.parent_child_rag import ParentChildRAG
from app.rag.rerank_rag import RerankRAG
from app.rag.service import RAGService
from app.retrieval.reranker import CrossEncoderReranker, LexicalReranker
from app.safety.medical_guardrails import MedicalGuardrails
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.document import Chunk, ChunkMetadata, SearchResult


def _chunk(
    chunk_id: str,
    text: str,
    *,
    section: str = "FAST",
    parent_id: str = "stroke-fast",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            doc_id="stroke",
            source_path="data/sample/stroke.md",
            source_type="md",
            title="Đột quỵ",
            created_at="2026-06-12T00:00:00Z",
            hash=chunk_id,
            section_title=section,
            parent_id=parent_id,
        ),
    )


class RecordingLLM:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[SearchResult]]] = []

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        self.calls.append((question, contexts))
        return "Câu trả lời có căn cứ [1]."


class RecordingRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results_by_query.get(query, [])[:top_k]


class AlwaysReturnsVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        deleted = set(chunk_ids)
        self.chunks = [chunk for chunk in self.chunks if chunk.chunk_id not in deleted]

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        return [SearchResult(chunk=chunk, score=0.9) for chunk in self.chunks[:top_k]]


class RecordingReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((query, len(results), top_k))
        return list(reversed(results))[:top_k]


class FakeTransformer:
    def __init__(self, expanded: list[str], rewritten: str = "") -> None:
        self.expanded = expanded
        self.rewritten = rewritten

    def expand(self, query: str) -> list[str]:
        return self.expanded

    def rewrite(self, query: str) -> str:
        return self.rewritten


class FakeParentStore:
    def __init__(self, expanded: list[SearchResult]) -> None:
        self.expanded = expanded
        self.calls: list[tuple[list[SearchResult], int]] = []

    def expand_parent_results(
        self,
        child_results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        self.calls.append((child_results, top_k))
        return self.expanded[:top_k]


class AdvancedRagTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guardrails = MedicalGuardrails()
        self.first = SearchResult(_chunk("stroke-1", "FAST gồm Face và Arm."), 0.8)
        self.second = SearchResult(_chunk("stroke-2", "Speech và Time hoàn thiện FAST."), 0.7)

    def test_rerank_retrieves_twenty_candidates_before_returning_top_k(self) -> None:
        retriever = RecordingRetriever({"FAST là gì?": [self.first, self.second]})
        reranker = RecordingReranker()
        llm = RecordingLLM()
        rag = RerankRAG(retriever, reranker, llm, self.guardrails)

        output = rag.answer("FAST là gì?", top_k=1)

        self.assertEqual(retriever.calls, [("FAST là gì?", 20)])
        self.assertEqual(reranker.calls, [("FAST là gì?", 2, 1)])
        self.assertEqual(output.results[0].chunk.chunk_id, "stroke-2")
        self.assertIn(MEDICAL_DISCLAIMER, output.answer)

    def test_parent_child_generates_from_expanded_parent_section(self) -> None:
        child_retriever = RecordingRetriever({"FAST": [self.first]})
        parent = SearchResult(
            _chunk(
                "parent-stroke-fast",
                "FAST gồm Face, Arm, Speech và Time. Cần gọi cấp cứu ngay.",
            ),
            0.8,
        )
        parent_store = FakeParentStore([parent])
        llm = RecordingLLM()
        rag = ParentChildRAG(child_retriever, parent_store, llm, self.guardrails)

        output = rag.answer("FAST", top_k=2)

        self.assertEqual(output.results, [parent])
        self.assertEqual(llm.calls[0][1], [parent])
        self.assertEqual(parent_store.calls[0][0], [self.first])

    def test_multi_query_fuses_results_from_all_unique_queries(self) -> None:
        retriever = RecordingRetriever(
            {
                "FAST": [self.first],
                "dấu hiệu đột quỵ": [self.second, self.first],
            }
        )
        transformer = FakeTransformer(["FAST", "dấu hiệu đột quỵ", "FAST"])
        rag = MultiQueryRAG(
            retriever,
            transformer,
            RecordingLLM(),
            self.guardrails,
        )

        results = rag.retrieve("Giải thích FAST", top_k=2)

        self.assertEqual(
            retriever.calls,
            [("FAST", 8), ("dấu hiệu đột quỵ", 8)],
        )
        self.assertEqual(
            {result.chunk.chunk_id for result in results},
            {"stroke-1", "stroke-2"},
        )

    def test_corrective_rewrites_once_when_initial_context_is_irrelevant(self) -> None:
        irrelevant = SearchResult(
            _chunk("lab-1", "Hemoglobin vận chuyển oxy.", section="Hemoglobin"),
            0.9,
        )
        retriever = RecordingRetriever(
            {
                "FAST?": [irrelevant],
                "FAST đột quỵ": [self.first],
            }
        )
        rag = CorrectiveRAG(
            retriever,
            FakeTransformer([], rewritten="FAST đột quỵ"),
            RecordingLLM(),
            self.guardrails,
        )

        output = rag.answer("FAST?", top_k=1)

        self.assertEqual(
            retriever.calls,
            [("FAST?", 1), ("FAST đột quỵ", 1)],
        )
        self.assertEqual(output.results, [self.first])

    def test_corrective_reports_insufficient_context_after_failed_retry(self) -> None:
        rag = CorrectiveRAG(
            RecordingRetriever({}),
            FakeTransformer([], rewritten="không có"),
            RecordingLLM(),
            self.guardrails,
        )

        output = rag.answer("Nội dung ngoài tài liệu", top_k=2)

        self.assertEqual(output.results, [])
        self.assertIn(INSUFFICIENT_CONTEXT_MESSAGE, output.answer)
        self.assertIn(MEDICAL_DISCLAIMER, output.answer)

    def test_service_supports_all_declared_rag_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                )
            )
            service.ingest_file(
                "stroke.md",
                "# FAST\n\nFAST gồm Face, Arm, Speech và Time.".encode(),
            )

            results = {
                mode: service.chat("FAST là gì?", rag_mode=mode, top_k=2)
                for mode in (
                    "naive",
                    "hybrid",
                    "rerank",
                    "parent_child",
                    "multi_query",
                    "corrective",
                    "agentic",
                )
            }

        self.assertTrue(all(result.sources for result in results.values()))
        self.assertEqual(set(results), {result.mode_used for result in results.values()})

    def test_service_agentic_mode_keeps_emergency_routing_before_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                )
            )

            result = service.chat(
                "Tôi đang đau ngực và khó thở, phải làm gì?",
                rag_mode="agentic",
            )

        self.assertEqual(result.mode_used, "agentic")
        self.assertEqual(result.safety_warning, "emergency")
        self.assertEqual(result.sources, [])

    def test_agentic_router_treats_stroke_comparison_as_educational_query(self) -> None:
        mode = RuleBasedAgenticRouter().choose_mode("So sánh FAST trong đột quỵ từ nhiều nguồn")

        self.assertEqual(mode, "multi_query")

    def test_service_agentic_retrieve_supports_educational_stroke_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                )
            )
            service.ingest_file(
                "stroke.md",
                "# FAST\n\nFAST gồm Face, Arm, Speech và Time.".encode(),
            )

            results = service.retrieve(
                "So sánh FAST trong đột quỵ từ nhiều nguồn",
                rag_mode="agentic",
                top_k=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.metadata.title, "stroke")

    def test_all_rag_modes_reject_unrelated_dense_results_before_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                ),
                vector_store=AlwaysReturnsVectorStore(),
            )
            service.ingest_file(
                "stroke.md",
                (
                    "# FAST\n\nFAST gồm Face, Arm, Speech và Time. "
                    "Hoạt động thể chất phù hợp hỗ trợ sức khỏe và giáo dục y khoa."
                ).encode(),
            )

            results = {
                mode: service.chat(
                    "Insulin hoạt động thế nào?",
                    rag_mode=mode,
                    top_k=2,
                )
                for mode in (
                    "naive",
                    "hybrid",
                    "rerank",
                    "parent_child",
                    "multi_query",
                    "corrective",
                    "agentic",
                )
            }

        self.assertTrue(all(not result.sources for result in results.values()))
        self.assertTrue(all(INSUFFICIENT_CONTEXT_MESSAGE in result.answer for result in results.values()))

    def test_lexical_reranker_remains_usable_as_default(self) -> None:
        reranked = LexicalReranker().rerank(
            "FAST Face Arm",
            [self.second, self.first],
            top_k=1,
        )

        self.assertEqual(reranked[0].chunk.chunk_id, "stroke-1")

    def test_lexical_reranker_rewards_topic_coverage_not_generic_overlap(self) -> None:
        generic = SearchResult(
            _chunk(
                "generic",
                "Metabolism is an important subject with many clinical roles.",
                section="Overview",
            ),
            0.9,
        )
        specific = SearchResult(
            _chunk(
                "specific",
                "Enzyme kinetics explains how enzymes regulate metabolic reactions.",
                section="Enzyme kinetics",
            ),
            0.6,
        )

        reranked = LexicalReranker().rerank(
            "enzyme kinetics metabolic reactions",
            [generic, specific],
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk.chunk_id, "specific")
        self.assertGreater(
            reranked[0].signals["lexical_coverage"],
            reranked[1].signals["lexical_coverage"],
        )

    def test_lexical_reranker_uses_dense_signal_for_cross_language_query(self) -> None:
        semantic = SearchResult(
            _chunk(
                "semantic",
                "Enzymes regulate metabolic reactions in cells.",
                section="Enzyme kinetics",
            ),
            0.45,
            {"semantic_score": 0.82, "dense_rank": 1.0},
        )
        lexical = SearchResult(
            _chunk(
                "lexical",
                "Chất xúc tác sinh học là một mục từ trong bảng thuật ngữ.",
                section="Glossary",
            ),
            0.9,
            {"semantic_score": 0.70, "dense_rank": 20.0},
        )

        reranked = LexicalReranker().rerank(
            "Chất xúc tác sinh học điều hòa phản ứng trong tế bào",
            [lexical, semantic],
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk.chunk_id, "semantic")
        self.assertEqual(reranked[0].signals["semantic_component"], 0.82)
        self.assertEqual(reranked[0].signals["dense_rank_component"], 1.0)

    def test_lexical_reranker_diversifies_near_duplicate_chunks(self) -> None:
        first = SearchResult(
            _chunk("enzyme-1", "Enzyme activity depends on pH and temperature."),
            0.9,
        )
        duplicate = SearchResult(
            _chunk("enzyme-2", "Enzyme activity depends on temperature and pH."),
            0.89,
        )
        complementary = SearchResult(
            _chunk("enzyme-3", "Competitive inhibitors increase apparent Km."),
            0.75,
        )

        reranked = LexicalReranker(diversity_lambda=0.55).rerank(
            "enzyme activity inhibitors",
            [first, duplicate, complementary],
            top_k=2,
        )

        self.assertEqual(
            {result.chunk.chunk_id for result in reranked},
            {"enzyme-1", "enzyme-3"},
        )

    def test_cross_encoder_reranker_uses_adapter_and_exposes_semantic_score(self) -> None:
        class FakeCrossEncoder:
            def predict(self, pairs, **kwargs):
                assert len(pairs) == 2
                return [-2.0, 3.0]

        reranker = CrossEncoderReranker(
            model_name="fake",
            model=FakeCrossEncoder(),
        )
        reranked = reranker.rerank(
            "enzyme regulation",
            [self.first, self.second],
            top_k=1,
        )

        self.assertEqual(reranked[0].chunk.chunk_id, "stroke-2")
        self.assertGreater(reranked[0].signals["semantic_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
