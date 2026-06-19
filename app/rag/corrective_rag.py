from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.naive_rag import RagOutput
from app.retrieval.relevance import filter_relevant_results
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


class CorrectiveRAG:
    def __init__(
        self,
        retriever,
        query_transformer,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
    ) -> None:
        self.retriever = retriever
        self.query_transformer = query_transformer
        self.llm = llm
        self.guardrails = guardrails

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        initial = self.retriever.retrieve(question, top_k=top_k)
        relevant = filter_relevant_results(question, initial)
        if relevant:
            return relevant

        rewritten = self.query_transformer.rewrite(question).strip()
        if not rewritten or rewritten == question.strip():
            return []
        retry = self.retriever.retrieve(rewritten, top_k=top_k)
        return filter_relevant_results(question, retry)

    def answer(self, question: str, top_k: int = 5) -> RagOutput:
        safety = self.guardrails.check(question)
        if not safety.allowed:
            return RagOutput(
                answer=safety.message or "",
                results=[],
                safety_warning=safety.safety_warning,
            )

        results = self.retrieve(question, top_k=top_k)
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
