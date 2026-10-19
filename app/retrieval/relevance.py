from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

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


@dataclass(frozen=True)
class RelevanceDecision:
    chunk_id: str
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)


class RelevancePolicy:
    def __init__(self, dense_threshold=0.45, reranker_threshold=0.5, lexical_coverage=0.5):
        self.dense_threshold = dense_threshold
        self.reranker_threshold = reranker_threshold
        self.lexical_coverage = lexical_coverage

    def evaluate(self, query: str, result: SearchResult) -> RelevanceDecision:
        terms = meaningful_query_terms(query)
        scores = dict(result.signals)
        if not terms:
            return RelevanceDecision(result.chunk.chunk_id, False, ["empty_topic"], scores)
        reranker = scores.get("reranker_score")
        if reranker is not None:
            accepted = math.isfinite(reranker) and reranker >= self.reranker_threshold
            return RelevanceDecision(
                result.chunk.chunk_id,
                accepted,
                ["reranker_accepted" if accepted else "reranker_below_threshold"],
                scores,
            )
        semantic = scores.get("dense_cosine", scores.get("semantic_score"))
        if semantic is not None and math.isfinite(semantic) and semantic >= self.dense_threshold:
            return RelevanceDecision(result.chunk.chunk_id, True, ["dense_accepted"], scores)
        searchable = " ".join(
            filter(None, (result.chunk.text, result.chunk.metadata.title, result.chunk.metadata.section_title))
        )
        identifiers = {word.lower() for word in re.findall(r"\b[A-Z]{2,}[A-Za-z0-9]*\b", query)}
        if identifiers & set(tokenize(searchable)):
            return RelevanceDecision(result.chunk.chunk_id, True, ["exact_identifier"], scores)
        title_terms = meaningful_query_terms(result.chunk.metadata.section_title or result.chunk.metadata.title)
        if len(title_terms) >= 2 and title_terms <= terms:
            return RelevanceDecision(result.chunk.chunk_id, True, ["specific_section_anchor"], scores)
        coverage = len(terms & set(tokenize(searchable))) / len(terms)
        scores["topic_coverage"] = coverage
        return RelevanceDecision(
            result.chunk.chunk_id,
            coverage >= self.lexical_coverage,
            ["lexical_accepted" if coverage >= self.lexical_coverage else "insufficient_topic_coverage"],
            scores,
        )


def filter_relevant_results(
    query: str,
    results: list[SearchResult],
    policy: RelevancePolicy | None = None,
) -> list[SearchResult]:
    policy = policy or RelevancePolicy()
    return [result for result in results if policy.evaluate(query, result).accepted]
