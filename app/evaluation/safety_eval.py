from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evaluation import EvalMetric


def evaluate_safety(service, dataset_path: str) -> list[EvalMetric]:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    safety_cases = [case for case in cases if case.get("expected") in {"emergency", "blocked_medical_advice"}]
    if not safety_cases:
        return [EvalMetric(name="safety_accuracy", value=0.0)]

    hits = 0
    for case in safety_cases:
        result = service.chat(case["question"], rag_mode="naive")
        hits += int(result.safety_warning == case["expected"])
    return [EvalMetric(name="safety_accuracy", value=hits / len(safety_cases))]
