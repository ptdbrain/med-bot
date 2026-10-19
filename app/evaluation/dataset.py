from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class EvalCase(BaseModel):
    case_id: str
    group_id: str
    split: Literal["dev", "test"]
    question: str
    query_language: str
    category: str
    expected: Literal["answerable", "insufficient_context", "emergency", "blocked_medical_advice"]
    relevant_evidence_ids: list[str]
    required_claims: list[str]
    unsupported_claims: list[str]
    reviewed_by: str | None = None
    provenance: str = "assistant_draft"


def load_cases(path: Path) -> list[EvalCase]:
    return [
        EvalCase.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_cases(cases: list[EvalCase], manifest: dict, *, require_review: bool = False) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    groups: dict[str, set[str]] = {}
    evidence_splits: dict[str, set[str]] = {}
    evidence = manifest.get("evidence", manifest)
    for case in cases:
        if case.case_id in seen:
            errors.append(f"{case.case_id}:duplicate_id")
        seen.add(case.case_id)
        groups.setdefault(case.group_id, set()).add(case.split)
        if case.expected == "answerable" and not case.relevant_evidence_ids:
            errors.append(f"{case.case_id}:missing_evidence")
        if require_review and not case.reviewed_by:
            errors.append(f"{case.case_id}:unreviewed")
        for item in case.relevant_evidence_ids:
            if item not in evidence:
                errors.append(f"{case.case_id}:unknown_evidence:{item}")
            evidence_splits.setdefault(item, set()).add(case.split)
    errors.extend(f"{group}:split_leakage" for group, splits in groups.items() if len(splits) > 1)
    errors.extend(f"{item}:evidence_leakage" for item, splits in evidence_splits.items() if len(splits) > 1)
    return errors
