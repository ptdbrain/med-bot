from __future__ import annotations

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock_llm import MockLLM
from app.llm.ollama_llm import OllamaLLM


def create_llm(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.strip().lower()
    if provider == "mock":
        return MockLLM()
    if provider == "ollama":
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    raise ValueError(
        f"LLM_PROVIDER='{settings.llm_provider}' không hợp lệ. "
        "Hỗ trợ: mock, ollama."
    )
