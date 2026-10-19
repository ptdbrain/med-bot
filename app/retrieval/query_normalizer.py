import unicodedata
from dataclasses import dataclass

from app.retrieval.vector_store import normalize_text


@dataclass(frozen=True)
class NormalizedQuery:
    original: str
    dense_text: str
    sparse_text: str
    language: str


def normalize_query(query: str) -> NormalizedQuery:
    dense = " ".join(unicodedata.normalize("NFC", query).split())
    folded = normalize_text(dense)
    language = "vi" if any(c in dense.lower() for c in "ăâđêôơưạảấầậểễộỡự") else "unknown"
    return NormalizedQuery(query, dense, folded, language)
