from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.naive_rag import RagOutput
from app.retrieval.relevance import filter_relevant_results
from app.safety.medical_guardrails import MedicalGuardrails
from app.schemas.document import SearchResult


class ParentChildRAG:
    def __init__(
        self,
        child_retriever,
        parent_store,
        llm: LLMProvider,
        guardrails: MedicalGuardrails,
    ) -> None:
        self.child_retriever = child_retriever
        self.parent_store = parent_store
        self.llm = llm
        self.guardrails = guardrails

    def retrieve(self, question: str, top_k: int = 5) -> list[SearchResult]:
        children = self.child_retriever.retrieve(question, top_k=top_k * 4)
        return self.parent_store.expand_parent_results(children, top_k=top_k)

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
