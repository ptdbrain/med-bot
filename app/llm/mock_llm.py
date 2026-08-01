from __future__ import annotations

from app.llm.base import LLMProvider
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.document import SearchResult

INSUFFICIENT_CONTEXT_MESSAGE = "Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp."


class MockLLM(LLMProvider):
    """Deterministic offline answer generator for portfolio demo and tests."""

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        if not contexts:
            return (
                f"{INSUFFICIENT_CONTEXT_MESSAGE}\n\n"
                "Nguồn đã dùng: Không có nguồn phù hợp.\n\n"
                f"Cảnh báo y tế: {MEDICAL_DISCLAIMER}"
            )

        source_lines: list[str] = []
        explanation_lines: list[str] = []
        for index, result in enumerate(contexts, start=1):
            text = " ".join(result.chunk.text.split())
            snippet = text[:420] + ("..." if len(text) > 420 else "")
            explanation_lines.append(f"[{index}] {snippet}")
            metadata = result.chunk.metadata
            location = metadata.section_title or metadata.title
            source_lines.append(f"[{index}] {metadata.title} - {location} ({metadata.source_path})")

        short_answer = (
            "Ở mức học tập, thông tin trong tài liệu gợi ý rằng câu trả lời nên dựa trên "
            f"{len(contexts)} nguồn được truy xuất liên quan đến câu hỏi."
        )
        return (
            "1. Câu trả lời ngắn gọn:\n"
            f"{short_answer}\n\n"
            "2. Giải thích chi tiết:\n"
            + "\n".join(explanation_lines)
            + "\n\n3. Nguồn đã dùng:\n"
            + "\n".join(source_lines)
            + "\n\n4. Cảnh báo y tế:\n"
            + MEDICAL_DISCLAIMER
        )
