from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.naive_rag import RagOutput
from app.rag.rag_fusion import reciprocal_rank_fusion
from app.retrieval.relevance import filter_relevant_results
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


class MultiQueryRAG:
    def __init__(
        self,
        retriever,
        query_transformer,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
        candidate_multiplier: int = 4,
    ) -> None:
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.llm = llm
        self.guardrails = guardrails
        self.candidate_multiplier = candidate_multiplier

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        queries = list(dict.fromkeys(self.query_transformer.expand(question)))
        if not queries:
            return []
        candidate_k = max(top_k, top_k * self.candidate_multiplier)
        result_lists = [
            self.retriever.retrieve(query, top_k=candidate_k)
            for query in queries
        ]
        return reciprocal_rank_fusion(result_lists)[:top_k]

    def answer(self, question: str, top_k: int = 5) -> RagOutput:
        safety = self.guardrails.check(question)
        if not safety.allowed:
            return RagOutput(
                answer=safety.message or "",
                results=[],
                safety_warning=safety.safety_warning,
            )

        results = filter_relevant_results(
            question,
            self.retrieve(question, top_k=top_k),
        )
        if not results:
            return RagOutput(
                answer=self.guardrails.with_disclaimer(INSUFFICIENT_CONTEXT_MESSAGE),
                results=[],
            )
        return RagOutput(
            answer=self.guardrails.with_disclaimer(
                self.llm.generate(question, results)
            ),
            results=results,
        )
