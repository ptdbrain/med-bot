from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evaluation import EvalMetric


def evaluate_retrieval(service, dataset_path: str, top_k: int = 5) -> list[EvalMetric]:
    cases = _load_cases(dataset_path)
    answerable = [case for case in cases if case.get("expected_source")]
    if not answerable:
        return [EvalMetric(name="hit_at_k", value=0.0), EvalMetric(name="mrr", value=0.0)]

    hits = 0
    reciprocal_ranks: list[float] = []
    for case in answerable:
        results = service.retrieve(case["question"], top_k=top_k)
        expected = case["expected_source"]
        rank = 0
        for index, result in enumerate(results, start=1):
            metadata = result.chunk.metadata
            if expected in metadata.source_path or expected in metadata.doc_id or expected in metadata.title:
                rank = index
                break
        hits += int(rank > 0)
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return [
        EvalMetric(name="hit_at_k", value=hits / len(answerable)),
        EvalMetric(name="mrr", value=sum(reciprocal_ranks) / len(reciprocal_ranks)),
    ]


def _load_cases(dataset_path: str) -> list[dict]:
    path = Path(dataset_path)
    return json.loads(path.read_text(encoding="utf-8"))
