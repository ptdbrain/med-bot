from __future__ import annotations

import json
from pathlib import Path

from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.evaluation import EvalMetric


def evaluate_answers(service, dataset_path: str) -> list[EvalMetric]:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not cases:
        return [EvalMetric(name="citation_rate", value=0.0), EvalMetric(name="disclaimer_rate", value=0.0)]

    citation_hits = 0
    disclaimer_hits = 0
    insufficient_hits = 0
    insufficient_cases = 0
    for case in cases:
        result = service.chat(case["question"], rag_mode=case.get("rag_mode", "naive"))
        citation_hits += int(bool(result.sources) or "Nguồn đã dùng" in result.answer)
        disclaimer_hits += int(MEDICAL_DISCLAIMER in result.answer)
        if case.get("expected") == "insufficient_context":
            insufficient_cases += 1
            insufficient_hits += int(INSUFFICIENT_CONTEXT_MESSAGE in result.answer)

    metrics = [
        EvalMetric(name="citation_rate", value=citation_hits / len(cases)),
        EvalMetric(name="disclaimer_rate", value=disclaimer_hits / len(cases)),
    ]
    if insufficient_cases:
        metrics.append(EvalMetric(name="insufficient_context_accuracy", value=insufficient_hits / insufficient_cases))
    return metrics
