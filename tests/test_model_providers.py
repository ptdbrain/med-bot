from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from app.config import Settings, get_settings
from app.llm.factory import create_llm
from app.llm.mock_llm import MockLLM
from app.llm.ollama_llm import OllamaLLM
from app.schemas.document import Chunk, ChunkMetadata, SearchResult


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


def _context() -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id="stroke-1",
            text="FAST gồm méo miệng, yếu tay, nói khó và thời gian gọi cấp cứu.",
            metadata=ChunkMetadata(
                doc_id="stroke",
                source_path="stroke.md",
                source_type="md",
                title="Hướng dẫn đột quỵ",
                created_at="2026-06-11T00:00:00Z",
                hash="abc",
                page_number=3,
                section_title="FAST",
            ),
        ),
        score=0.91,
    )


class ModelProviderTests(unittest.TestCase):
    def test_model_settings_are_read_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "ollama",
                "OLLAMA_BASE_URL": "http://host.test:11434/",
                "OLLAMA_MODEL": "qwen3:4b",
                "OLLAMA_TIMEOUT_SECONDS": "45",
                "OLLAMA_THINK": "true",
                "OLLAMA_NUM_PREDICT": "128",
                "EMBEDDING_PROVIDER": "sentence-transformers",
                "EMBEDDING_MODEL": "BAAI/bge-m3",
                "EMBEDDING_DEVICE": "cpu",
                "EMBEDDING_DIMENSION": "1024",
                "EMBEDDING_BATCH_SIZE": "4",
                "VECTOR_STORE_PROVIDER": "qdrant",
                "QDRANT_COLLECTION": "medical_chunks",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.ollama_base_url, "http://host.test:11434")
        self.assertEqual(settings.ollama_model, "qwen3:4b")
        self.assertEqual(settings.ollama_timeout_seconds, 45)
        self.assertTrue(settings.ollama_think)
        self.assertEqual(settings.ollama_num_predict, 128)
        self.assertEqual(settings.embedding_provider, "sentence-transformers")
        self.assertEqual(settings.embedding_model, "BAAI/bge-m3")
        self.assertEqual(settings.embedding_dimension, 1024)
        self.assertEqual(settings.embedding_batch_size, 4)
        self.assertEqual(settings.vector_store_provider, "qdrant")
        self.assertEqual(settings.qdrant_collection, "medical_chunks")

    def test_ollama_sends_grounded_non_streaming_chat_request(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "message": {
                        "role": "assistant",
                        "content": "FAST gồm bốn dấu hiệu chính [1].",
                    }
                }
            )
        )
        provider = OllamaLLM(
            base_url="http://localhost:11434/",
            model="qwen3:4b",
            timeout_seconds=30,
            think=False,
            num_predict=128,
            session=session,
        )

        answer = provider.generate("FAST gồm gì?", [_context()])

        self.assertEqual(answer, "FAST gồm bốn dấu hiệu chính [1].")
        self.assertEqual(session.calls[0]["url"], "http://localhost:11434/api/chat")
        self.assertEqual(session.calls[0]["timeout"], 30)
        payload = session.calls[0]["json"]
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertFalse(payload["stream"])
        self.assertTrue(payload["think"])
        self.assertEqual(payload["options"]["num_predict"], 128)
        self.assertIn("[1]", payload["messages"][1]["content"])
        self.assertIn("Hướng dẫn đột quỵ", payload["messages"][1]["content"])
        self.assertTrue(payload["messages"][1]["content"].endswith("\n/no_think"))
        self.assertIn("không đủ thông tin", payload["messages"][0]["content"])
        for section in (
            "1. Câu trả lời ngắn gọn",
            "2. Giải thích chi tiết",
            "3. Nguồn đã dùng",
            "4. Cảnh báo y tế",
        ):
            self.assertIn(section, payload["messages"][0]["content"])

    def test_ollama_rejects_empty_answer(self) -> None:
        provider = OllamaLLM(
            base_url="http://localhost:11434",
            model="qwen3:4b",
            session=FakeSession(FakeResponse({"message": {"content": "  "}})),
        )

        with self.assertRaisesRegex(RuntimeError, "không trả về nội dung"):
            provider.generate("FAST gồm gì?", [_context()])

    def test_llm_factory_preserves_mock_default(self) -> None:
        self.assertIsInstance(create_llm(Settings()), MockLLM)

    def test_llm_factory_builds_ollama_provider(self) -> None:
        provider = create_llm(
            Settings(
                llm_provider="ollama",
                ollama_base_url="http://ollama.test:11434",
                ollama_model="qwen3:4b",
                ollama_think=True,
                ollama_num_predict=96,
            )
        )

        self.assertIsInstance(provider, OllamaLLM)
        self.assertEqual(provider.base_url, "http://ollama.test:11434")
        self.assertTrue(provider.think)
        self.assertEqual(provider.num_predict, 96)

    def test_ollama_thinking_mode_does_not_add_no_think_directive(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "message": {
                        "content": "FAST gồm bốn dấu hiệu chính [1].",
                    }
                }
            )
        )
        provider = OllamaLLM(
            base_url="http://localhost:11434",
            think=True,
            session=session,
        )

        provider.generate("FAST gồm gì?", [_context()])

        payload = session.calls[0]["json"]
        self.assertTrue(payload["think"])
        self.assertNotIn("/no_think", payload["messages"][1]["content"])

    def test_non_reasoning_model_keeps_thinking_parser_disabled(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "message": {
                        "content": "Grounded answer [1].",
                    }
                }
            )
        )
        provider = OllamaLLM(
            base_url="http://localhost:11434",
            model="llama3.2:3b",
            think=False,
            session=session,
        )

        provider.generate("Question?", [_context()])

        self.assertFalse(session.calls[0]["json"]["think"])

    def test_qwen_instruct_model_does_not_enable_hybrid_thinking(self) -> None:
        session = FakeSession(
            FakeResponse(
                {
                    "message": {
                        "content": "Grounded answer [1].",
                    }
                }
            )
        )
        provider = OllamaLLM(
            base_url="http://localhost:11434",
            model="qwen3:4b-instruct-2507-q4_K_M",
            think=False,
            session=session,
        )

        provider.generate("Question?", [_context()])

        payload = session.calls[0]["json"]
        self.assertFalse(payload["think"])
        self.assertNotIn("/no_think", payload["messages"][1]["content"])

    def test_llm_factory_rejects_unknown_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
            create_llm(Settings(llm_provider="unknown"))


if __name__ == "__main__":
    unittest.main()
