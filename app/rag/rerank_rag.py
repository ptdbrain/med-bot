from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.naive_rag import RagOutput
from app.retrieval.relevance import filter_relevant_results
from app.retrieval.reranker import Reranker
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


class RerankRAG:
    def __init__(
        self,
        retriever,
        reranker: Reranker,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
        candidate_k: int = 20,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.guardrails = guardrails
        self.candidate_k = candidate_k

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []
        candidate_k = max(self.candidate_k, top_k)
        if hasattr(self.retriever, "retrieve_candidates"):
            candidates = self.retriever.retrieve_candidates(question)[:candidate_k]
        else:
            candidates = self.retriever.retrieve(question, top_k=candidate_k)
        return self.reranker.rerank(question, candidates, top_k=top_k)

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
            answer=self.guardrails.with_disclaimer(self.llm.generate(question, results)),
            results=results,
        )
