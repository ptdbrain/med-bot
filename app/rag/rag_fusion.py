from __future__ import annotations

from app.schemas.document import SearchResult


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    weights: list[float] | None = None,
    source_names: list[str] | None = None,
) -> list[SearchResult]:
    if k <= 0:
        raise ValueError("RRF k must be positive")
    if weights is None:
        weights = [1.0] * len(result_lists)
    if source_names is None:
        source_names = [f"source_{index}" for index in range(len(result_lists))]
    if len(weights) != len(result_lists):
        raise ValueError("RRF weights must match result lists")
    if any(weight < 0 for weight in weights):
        raise ValueError("RRF weights cannot be negative")
    if len(source_names) != len(result_lists):
        raise ValueError("RRF source names must match result lists")

    scores: dict[str, float] = {}
    chunks = {}
    signals: dict[str, dict[str, float]] = {}
    for results, weight, source_name in zip(
        result_lists,
        weights,
        source_names,
        strict=True,
    ):
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank)
            chunks[chunk_id] = result.chunk
            result_signals = signals.setdefault(chunk_id, {})
            result_signals.update(result.signals)
            result_signals[f"{source_name}_rank"] = float(rank)
            result_signals[f"{source_name}_score"] = float(result.score)
    # Python's sort is stable, so equal fusion scores preserve source priority.
    # Hybrid retrieval passes dense results before sparse results.
    ranked_ids = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
    return [
        SearchResult(
            chunk=chunks[chunk_id],
            score=round(scores[chunk_id], 6),
            signals=signals[chunk_id],
        )
        for chunk_id in ranked_ids
    ]
