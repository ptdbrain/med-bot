from __future__ import annotations

from app.llm.base import LLMProvider
from app.schemas.document import SearchResult


class OpenAILLM(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        raise NotImplementedError("OpenAI provider boundary is ready, but Phase 1 runs in mock mode by default.")
