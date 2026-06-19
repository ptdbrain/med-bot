from __future__ import annotations

from dataclasses import dataclass

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.retrieval.relevance import filter_relevant_results
from app.retrieval.vector_store import VectorStore
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


@dataclass(frozen=True)
class RagOutput:
    answer: str
    results: list[SearchResult]
    safety_warning: str | None = None


class NaiveRAG:
    def __init__(
        self,
        vector_store: VectorStore,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
        min_score: float = 0.08,
    ) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.guardrails = guardrails
        self.min_score = min_score

    def answer(self, question: str, top_k: int = 5) -> RagOutput:
        safety = self.guardrails.check(question)
        if not safety.allowed:
            return RagOutput(answer=safety.message or "", results=[], safety_warning=safety.safety_warning)

        results = filter_relevant_results(
            question,
            self.vector_store.search(
                question,
                top_k=top_k,
                min_score=self.min_score,
            ),
        )
        if not results:
            answer = self.guardrails.with_disclaimer(INSUFFICIENT_CONTEXT_MESSAGE)
            return RagOutput(answer=answer, results=[])

        answer = self.guardrails.with_disclaimer(self.llm.generate(question, results))
        return RagOutput(answer=answer, results=results)
