from __future__ import annotations

import json
import re
from pathlib import Path

from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.evaluation import EvalMetric

REQUIRED_ANSWER_SECTIONS = (
    "1. Câu trả lời ngắn gọn",
    "2. Giải thích chi tiết",
    "3. Nguồn đã dùng",
    "4. Cảnh báo y tế",
)


def has_required_answer_format(answer: str) -> bool:
    if all(section in answer for section in REQUIRED_ANSWER_SECTIONS):
        return True
    return all(re.search(rf"(?m)^\s*{index}\.\s+\S", answer) for index in range(1, 5))


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
                EvalMetric(name="format_compliance_rate", value=0.0),
                EvalMetric(name="disclaimer_rate", value=0.0),
            ],
            [],
        )

    citation_hits = 0
    citation_cases = 0
    format_hits = 0
    disclaimer_hits = 0
    insufficient_hits = 0
    insufficient_cases = 0
    details: list[dict] = []
    for case in cases:
        mode = case.get("rag_mode", "naive")
        result = service.chat(case["question"], rag_mode=mode)
        citation_ok = bool(result.sources) and bool(
            re.search(r"\[\d+\]", result.answer),
        )
        format_ok = has_required_answer_format(result.answer)
        disclaimer_ok = MEDICAL_DISCLAIMER in result.answer
        if case.get("expected") == "answerable":
            citation_cases += 1
            citation_hits += int(citation_ok)
            format_hits += int(format_ok)
        disclaimer_hits += int(disclaimer_ok)
        insufficient_ok: bool | None = None
        if case.get("expected") == "insufficient_context":
            insufficient_cases += 1
            insufficient_ok = INSUFFICIENT_CONTEXT_MESSAGE in result.answer
            insufficient_hits += int(insufficient_ok)
        answerable_ok = citation_ok and format_ok if case.get("expected") == "answerable" else True
        details.append(
            {
                "category": "answer",
                "question": case["question"],
                "mode": mode,
                "expected": case.get("expected"),
                "source_count": len(result.sources),
                "citation_ok": citation_ok,
                "format_ok": format_ok,
                "disclaimer_ok": disclaimer_ok,
                "insufficient_context_ok": insufficient_ok,
                "passed": disclaimer_ok and answerable_ok and (insufficient_ok is not False),
            }
        )

    metrics = [
        EvalMetric(
            name="citation_rate",
            value=citation_hits / citation_cases if citation_cases else 0.0,
        ),
        EvalMetric(
            name="format_compliance_rate",
            value=format_hits / citation_cases if citation_cases else 0.0,
        ),
        EvalMetric(name="disclaimer_rate", value=disclaimer_hits / len(cases)),
    ]
    if insufficient_cases:
        metrics.append(EvalMetric(name="insufficient_context_accuracy", value=insufficient_hits / insufficient_cases))
    return metrics, details
