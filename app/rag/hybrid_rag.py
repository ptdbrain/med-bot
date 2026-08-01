from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.naive_rag import RagOutput
from app.rag.rag_fusion import reciprocal_rank_fusion
from app.retrieval.bm25_store import BM25Store
from app.retrieval.relevance import filter_relevant_results
from app.retrieval.vector_store import VectorStore
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


class HybridRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_store: BM25Store,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.llm = llm
        self.guardrails = guardrails
        self.candidate_multiplier = candidate_multiplier
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        dense_results = self.vector_store.search(question, top_k=candidate_k, min_score=0.0)
        sparse_results = self.bm25_store.search(question, top_k=candidate_k)
        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            weights=[self.dense_weight, self.sparse_weight],
            source_names=["dense", "sparse"],
        )[:top_k]

    def answer(self, question: str, top_k: int = 5) -> RagOutput:
        safety = self.guardrails.check(question)
        if not safety.allowed:
            return RagOutput(answer=safety.message or "", results=[], safety_warning=safety.safety_warning)

        results = filter_relevant_results(
            question,
            self.retrieve(question, top_k=top_k),
        )
        if not results:
            answer = self.guardrails.with_disclaimer(INSUFFICIENT_CONTEXT_MESSAGE)
            return RagOutput(answer=answer, results=[])

        answer = self.guardrails.with_disclaimer(self.llm.generate(question, results))
        return RagOutput(answer=answer, results=results)
