from __future__ import annotations

from app.llm.base import LLMProvider
from app.llm.mock_llm import MockLLM
from app.schemas.document import SearchResult


class LocalLLM(LLMProvider):
    def __init__(self) -> None:
        self._fallback = MockLLM()

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        return self._fallback.generate(question, contexts)
