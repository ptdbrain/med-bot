from __future__ import annotations

from app.retrieval.vector_store import tokenize
from app.schemas.document import SearchResult

QUESTION_STOPWORDS = {
    "ai",
    "cai",
    "cau",
    "cho",
    "co",
    "duoc",
    "does",
    "dong",
    "dung",
    "education",
    "explain",
    "for",
    "gi",
    "giup",
    "gom",
    "hay",
    "hoat",
    "hoi",
    "how",
    "in",
    "is",
    "khong",
    "la",
    "lieu",
    "mot",
    "nao",
    "nay",
    "nghia",
    "nhung",
    "noi",
    "tai",
    "the",
    "thong",
    "tin",
    "toi",
    "tro",
    "trong",
    "vai",
    "ve",
    "xin",
    "what",
}
SEMANTIC_RELEVANCE_THRESHOLD = 0.45


def meaningful_query_terms(query: str) -> set[str]:
    return set(tokenize(query)) - QUESTION_STOPWORDS


def filter_relevant_results(
    query: str,
    results: list[SearchResult],
) -> list[SearchResult]:
    query_terms = meaningful_query_terms(query)
    if not query_terms:
        return []

    relevant: list[SearchResult] = []
    for result in results:
        semantic_score = result.signals.get("semantic_score")
        if semantic_score is not None and semantic_score >= SEMANTIC_RELEVANCE_THRESHOLD:
            relevant.append(result)
            continue
        metadata = result.chunk.metadata
        searchable = " ".join(
            filter(
                None,
                (
                    result.chunk.text,
                    metadata.title,
                    metadata.section_title,
                    metadata.source_path,
                ),
            )
        )
        overlap = query_terms.intersection(tokenize(searchable))
        if overlap:
            relevant.append(result)
    return relevant
