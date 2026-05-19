from __future__ import annotations

from app.retrieval.vector_store import normalize_text


class RuleBasedAgenticRouter:
    def choose_mode(self, question: str) -> str:
        normalized = normalize_text(question)
        if any(term in normalized for term in {"cap cuu", "dau nguc", "kho tho", "dot quy"}):
            return "safety"
        if any(term in normalized for term in {"anh", "hinh", "infographic", "xet nghiem"}):
            return "image"
        if any(term in normalized for term in {"so sanh", "nhieu nguon", "tong hop"}):
            return "multi_query"
        return "naive"
