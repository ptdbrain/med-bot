from __future__ import annotations

from typing import Protocol

import requests

from app.llm.base import LLMProvider
from app.schemas.document import SearchResult

SYSTEM_PROMPT = """Bạn là trợ lý học tập y khoa dựa trên tài liệu.
Chỉ trả lời bằng thông tin có trong ngữ cảnh được cung cấp.
Mỗi nhận định phải gắn citation dạng [1], [2] tương ứng với nguồn.
Nếu ngữ cảnh không đủ thông tin, hãy nói rõ tài liệu được cung cấp không đủ thông tin.
Không chẩn đoán, kê đơn hoặc đưa ra quyết định điều trị cá nhân.
Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn theo đúng bốn phần:
1. Câu trả lời ngắn gọn
2. Giải thích chi tiết
3. Nguồn đã dùng
4. Cảnh báo y tế
Phần 4 phải có nguyên văn: Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ."""


class HttpSession(Protocol):
    def post(self, url: str, **kwargs): ...


class OllamaLLM(LLMProvider):
    def __init__(
        self,
        base_url: str,
        model: str = "qwen3:4b-instruct-2507-q4_K_M",
        timeout_seconds: int = 180,
        think: bool = False,
        num_predict: int = 512,
        session: HttpSession | None = None,
    ) -> None:
        if num_predict <= 0:
            raise ValueError("num_predict must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.think = think
        self.num_predict = num_predict
        self.session = session or requests.Session()

    def generate(self, question: str, contexts: list[SearchResult]) -> str:
        hybrid_thinking_model = "qwen3" in self.model.lower() and "instruct" not in self.model.lower()
        # Qwen3 templates emit <think> even when reasoning is discouraged.
        # Keeping the API parser enabled prevents that text leaking into content.
        parse_thinking = self.think or hybrid_thinking_model
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        question,
                        contexts,
                        disable_thinking=not self.think and hybrid_thinking_model,
                    ),
                },
            ],
            "stream": False,
            "think": parse_thinking,
            "options": {
                "temperature": 0.1,
                "num_predict": self.num_predict,
            },
        }
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise RuntimeError(f"Không thể gọi Ollama model '{self.model}' tại {self.base_url}: {exc}") from exc

        content = str(body.get("message", {}).get("content", "")).strip()
        if not content:
            raise RuntimeError(f"Ollama model '{self.model}' không trả về nội dung.")
        return content

    def _build_user_prompt(
        self,
        question: str,
        contexts: list[SearchResult],
        *,
        disable_thinking: bool = False,
    ) -> str:
        source_blocks: list[str] = []
        for index, result in enumerate(contexts, start=1):
            metadata = result.chunk.metadata
            location = metadata.section_title or metadata.title
            if metadata.page_number is not None:
                location = f"{location}, trang {metadata.page_number}"
            source_blocks.append(f"[{index}] {metadata.title} - {location}\n{result.chunk.text}")
        context_text = "\n\n".join(source_blocks)
        prompt = (
            f"Câu hỏi: {question}\n\n"
            f"Ngữ cảnh:\n{context_text}\n\n"
            "Hãy trả lời chỉ dựa trên ngữ cảnh và giữ nguyên số citation."
        )
        if disable_thinking:
            return f"{prompt}\n/no_think"
        return prompt
