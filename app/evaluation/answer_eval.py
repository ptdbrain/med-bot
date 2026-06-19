from __future__ import annotations

import json
from pathlib import Path

from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.evaluation import EvalMetric


def evaluate_answers(service, dataset_path: str) -> list[EvalMetric]:
    metrics, _ = evaluate_answers_with_details(service, dataset_path)
    return metrics


def evaluate_answers_with_details(
    service,
    dataset_path: str,
) -> tuple[list[EvalMetric], list[dict]]:
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not cases:
        return (
            [
                EvalMetric(name="citation_rate", value=0.0),
                EvalMetric(name="disclaimer_rate", value=0.0),
            ],
            [],
        )

    citation_hits = 0
    citation_cases = 0
    disclaimer_hits = 0
    insufficient_hits = 0
    insufficient_cases = 0
    details: list[dict] = []
    for case in cases:
        mode = case.get("rag_mode", "naive")
        result = service.chat(case["question"], rag_mode=mode)
        citation_ok = bool(result.sources) or "Nguồn đã dùng" in result.answer
        disclaimer_ok = MEDICAL_DISCLAIMER in result.answer
        if case.get("expected") == "answerable":
            citation_cases += 1
            citation_hits += int(citation_ok)
        disclaimer_hits += int(disclaimer_ok)
        insufficient_ok: bool | None = None
        if case.get("expected") == "insufficient_context":
            insufficient_cases += 1
            insufficient_ok = INSUFFICIENT_CONTEXT_MESSAGE in result.answer
            insufficient_hits += int(insufficient_ok)
        answerable_ok = (
            citation_ok if case.get("expected") == "answerable" else True
        )
        details.append(
            {
                "category": "answer",
                "question": case["question"],
                "mode": mode,
                "expected": case.get("expected"),
                "source_count": len(result.sources),
                "citation_ok": citation_ok,
                "disclaimer_ok": disclaimer_ok,
                "insufficient_context_ok": insufficient_ok,
                "passed": disclaimer_ok
                and answerable_ok
                and (insufficient_ok is not False),
            }
        )

    metrics = [
        EvalMetric(
            name="citation_rate",
            value=citation_hits / citation_cases if citation_cases else 0.0,
        ),
        EvalMetric(name="disclaimer_rate", value=disclaimer_hits / len(cases)),
    ]
    if insufficient_cases:
        metrics.append(EvalMetric(name="insufficient_context_accuracy", value=insufficient_hits / insufficient_cases))
    return metrics, details
