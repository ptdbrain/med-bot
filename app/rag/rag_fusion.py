from __future__ import annotations


def reciprocal_rank_fusion(result_lists: list[list], k: int = 60) -> list:
    scores: dict[str, float] = {}
    objects: dict[str, object] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            objects[chunk_id] = result
    return [objects[chunk_id] for chunk_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
