from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evaluation import EvalMetric


def evaluate_safety(service, dataset_path: str) -> list[EvalMetric]:
    metrics, _ = evaluate_safety_with_details(service, dataset_path)
    return metrics


def evaluate_safety_with_details(
    service,
    dataset_path: str,
) -> tuple[list[EvalMetric], list[dict]]:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    safety_cases = [case for case in cases if case.get("expected") in {"emergency", "blocked_medical_advice"}]
    if not safety_cases:
        return [EvalMetric(name="safety_accuracy", value=0.0)], []

    hits = 0
    details: list[dict] = []
    for case in safety_cases:
        mode = case.get("rag_mode", "agentic")
        result = service.chat(case["question"], rag_mode=mode)
        passed = result.safety_warning == case["expected"]
        hits += int(passed)
        details.append(
            {
                "category": "safety",
                "question": case["question"],
                "mode": mode,
                "expected": case["expected"],
                "actual_warning": result.safety_warning,
                "passed": passed,
            }
        )
    return (
        [
            EvalMetric(
                name="safety_accuracy",
                value=hits / len(safety_cases),
            )
        ],
        details,
    )
