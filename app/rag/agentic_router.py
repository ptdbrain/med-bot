from __future__ import annotations

from app.retrieval.vector_store import normalize_text


def _contains_term(text: str, term: str) -> bool:
    return f" {term} " in f" {text} "


class RuleBasedAgenticRouter:
    def choose_mode(self, question: str) -> str:
        normalized = normalize_text(question)
        if any(_contains_term(normalized, term) for term in ("anh", "hinh", "infographic", "xet nghiem")):
            return "hybrid"
        if any(_contains_term(normalized, term) for term in ("so sanh", "nhieu nguon", "tong hop")):
            return "multi_query"
        if any(_contains_term(normalized, term) for term in ("mo rong", "tim lai", "khong ro")):
            return "corrective"
        return "naive"
