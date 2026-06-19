from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.vector_store import normalize_text
from app.safety.policy import EMERGENCY_RESPONSE, MEDICAL_DISCLAIMER


DIRECT_EMERGENCY_KEYWORDS = {
    "cap cuu",
    "dau nguc",
    "kho tho",
    "meo mieng",
    "yeu tay",
    "mat y thuc",
    "ngat",
    "chay mau nang",
    "tu tu",
    "tu sat",
    "co giat",
    "chest pain",
    "shortness of breath",
    "unconscious",
    "suicide",
}

CONTEXTUAL_EMERGENCY_KEYWORDS = {
    "dot quy",
    "stroke",
}

PERSONAL_URGENCY_MARKERS = {
    "toi bi",
    "toi dang bi",
    "nguoi nha toi",
    "dang co",
    "nghi bi",
    "co phai toi",
    "can lam gi",
    "phai lam gi",
    "luc nay",
    "ngay bay gio",
}


@dataclass(frozen=True)
class EmergencyCheck:
    is_emergency: bool
    message: str | None = None
    matched_terms: list[str] | None = None


class EmergencyDetector:
    def check(self, text: str) -> EmergencyCheck:
        normalized = normalize_text(text)
        matched = sorted(term for term in DIRECT_EMERGENCY_KEYWORDS if term in normalized)
        contextual = [term for term in CONTEXTUAL_EMERGENCY_KEYWORDS if term in normalized]
        has_urgent_context = any(marker in normalized for marker in PERSONAL_URGENCY_MARKERS)
        if contextual and has_urgent_context:
            matched.extend(contextual)
        if not matched:
            return EmergencyCheck(is_emergency=False, matched_terms=[])
        return EmergencyCheck(
            is_emergency=True,
            matched_terms=matched,
            message=f"{EMERGENCY_RESPONSE} {MEDICAL_DISCLAIMER}",
        )
