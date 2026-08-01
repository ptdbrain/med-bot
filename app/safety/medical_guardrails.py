from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.vector_store import normalize_text
from app.safety.emergency_detector import EmergencyDetector
from app.safety.policy import MEDICAL_DISCLAIMER, PROHIBITED_MEDICAL_RESPONSE

DIAGNOSIS_KEYWORDS = {
    "chan doan",
    "toi bi benh gi",
    "benh cua toi",
    "ket luan benh",
    "diagnose",
}

PRESCRIPTION_KEYWORDS = {
    "ke don",
    "toa thuoc",
    "nen uong thuoc gi",
    "lieu dung",
    "prescribe",
    "dosage",
}


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    message: str | None = None
    safety_warning: str | None = None


class MedicalGuardrails:
    def __init__(self, emergency_detector: EmergencyDetector | None = None) -> None:
        self.emergency_detector = emergency_detector or EmergencyDetector()

    def check(self, text: str) -> GuardrailResult:
        emergency = self.emergency_detector.check(text)
        if emergency.is_emergency:
            return GuardrailResult(
                allowed=False,
                message=emergency.message,
                safety_warning="emergency",
            )

        normalized = normalize_text(text)
        asks_diagnosis = any(term in normalized for term in DIAGNOSIS_KEYWORDS)
        asks_prescription = any(term in normalized for term in PRESCRIPTION_KEYWORDS)
        if asks_diagnosis or asks_prescription:
            return GuardrailResult(
                allowed=False,
                message=f"{PROHIBITED_MEDICAL_RESPONSE} {MEDICAL_DISCLAIMER}",
                safety_warning="blocked_medical_advice",
            )
        return GuardrailResult(allowed=True)

    def with_disclaimer(self, answer: str) -> str:
        if MEDICAL_DISCLAIMER in answer:
            return answer
        return f"{answer}\n\nCảnh báo y tế: {MEDICAL_DISCLAIMER}"
