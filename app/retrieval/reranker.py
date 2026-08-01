from __future__ import annotations

import math
from typing import Any, Protocol

from app.retrieval.relevance import meaningful_query_terms
from app.retrieval.vector_store import tokenize
from app.schemas.document import SearchResult


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]: ...


def _context_text(result: SearchResult) -> str:
    metadata = result.chunk.metadata
    return "\n".join(
        value
        for value in (
            metadata.title,
            metadata.section_title,
            result.chunk.text,
        )
        if value
    )


def _jaccard_similarity(left: SearchResult, right: SearchResult) -> float:
    left_terms = set(tokenize(left.chunk.text))
    right_terms = set(tokenize(right.chunk.text))
    union = left_terms | right_terms
    if not union:
        return 0.0
    return len(left_terms & right_terms) / len(union)


def _select_diverse(
    results: list[SearchResult],
    top_k: int,
    diversity_lambda: float,
) -> list[SearchResult]:
    if top_k <= 0 or not results:
        return []
    if not 0.0 <= diversity_lambda <= 1.0:
        raise ValueError("diversity_lambda must be between 0 and 1")

    remaining = list(results)
    selected: list[SearchResult] = []
    maximum_score = max(result.score for result in remaining) or 1.0
    while remaining and len(selected) < top_k:
        if not selected:
            chosen = max(remaining, key=lambda item: item.score)
        else:
            chosen = max(
                remaining,
                key=lambda item: (
                    diversity_lambda * (item.score / maximum_score)
                    - (1.0 - diversity_lambda) * max(_jaccard_similarity(item, prior) for prior in selected)
                ),
            )
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _scalar_score(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("Cross-encoder must return one score per pair")
        return float(value[0])
    return float(value)


class LexicalReranker:
    """Deterministic multi-signal reranker for offline and test environments."""

    def __init__(self, diversity_lambda: float = 0.75) -> None:
        self.diversity_lambda = diversity_lambda

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []

        query_terms = meaningful_query_terms(query)
        maximum_retrieval_score = max(result.score for result in results) or 1.0
        has_semantic_scores = any("semantic_score" in result.signals for result in results)
        rescored: list[SearchResult] = []
        normalized_query = " ".join(tokenize(query))

        for result in results:
            context_terms = set(tokenize(_context_text(result)))
            title_terms = set(
                tokenize(
                    " ".join(
                        filter(
                            None,
                            (
                                result.chunk.metadata.title,
                                result.chunk.metadata.section_title,
                            ),
                        )
                    )
                )
            )
            overlap = query_terms & context_terms
            coverage = len(overlap) / max(1, len(query_terms))
            title_coverage = len(query_terms & title_terms) / max(
                1,
                len(query_terms),
            )
            normalized_context = " ".join(tokenize(_context_text(result)))
            phrase_match = float(bool(normalized_query) and normalized_query in normalized_context)
            retrieval_score = result.score / maximum_retrieval_score
            semantic_score = max(
                0.0,
                min(1.0, result.signals.get("semantic_score", 0.0)),
            )
            dense_rank = result.signals.get("dense_rank")
            dense_rank_component = 1.0 / math.sqrt(max(1.0, dense_rank)) if dense_rank is not None else 0.0
            if has_semantic_scores:
                score = (
                    0.15 * retrieval_score
                    + 0.40 * semantic_score
                    + 0.20 * dense_rank_component
                    + 0.17 * coverage
                    + 0.06 * title_coverage
                    + 0.02 * phrase_match
                )
            else:
                score = 0.40 * retrieval_score + 0.40 * coverage + 0.15 * title_coverage + 0.05 * phrase_match
            signals = {
                **result.signals,
                "retrieval_score": float(result.score),
                "semantic_component": round(semantic_score, 6),
                "dense_rank_component": round(dense_rank_component, 6),
                "lexical_coverage": round(coverage, 6),
                "title_coverage": round(title_coverage, 6),
                "phrase_match": phrase_match,
            }
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=round(score, 6),
                    signals=signals,
                )
            )

        rescored.sort(key=lambda item: item.score, reverse=True)
        return _select_diverse(
            rescored,
            top_k=top_k,
            diversity_lambda=self.diversity_lambda,
        )


class CrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder adapter with diversity selection."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 8,
        diversity_lambda: float = 0.75,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.diversity_lambda = diversity_lambda
        self._model = model

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not results:
            return []
        pairs = [[query, _context_text(result)] for result in results]
        raw_scores = self._get_model().predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        rescored: list[SearchResult] = []
        for result, raw_score in zip(results, raw_scores, strict=True):
            logit = _scalar_score(raw_score)
            semantic_score = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
            rescored.append(
                SearchResult(
                    chunk=result.chunk,
                    score=round(semantic_score, 6),
                    signals={
                        **result.signals,
                        "reranker_logit": logit,
                        "semantic_score": round(semantic_score, 6),
                    },
                )
            )
        rescored.sort(key=lambda item: item.score, reverse=True)
        return _select_diverse(
            rescored,
            top_k=top_k,
            diversity_lambda=self.diversity_lambda,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Cross-encoder reranking requires sentence-transformers. "
                "Install it with: pip install -r requirements-rag.txt"
            ) from exc
        self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model
