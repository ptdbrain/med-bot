from __future__ import annotations

from abc import ABC, abstractmethod

from app.schemas.document import SearchResult


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        raise NotImplementedError
