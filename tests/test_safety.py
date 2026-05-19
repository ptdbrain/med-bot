from app.safety.emergency_detector import EmergencyDetector
from app.safety.medical_guardrails import MedicalGuardrails
from app.safety.policy import MEDICAL_DISCLAIMER
from app.retrieval.vector_store import normalize_text


def test_emergency_detector_recognizes_dangerous_symptoms():
    detector = EmergencyDetector()

    result = detector.check("Toi bi dau nguc va kho tho rat nhieu")

    assert result.is_emergency is True
    assert "cap cuu" in normalize_text(result.message or "")


def test_emergency_detector_allows_educational_stroke_question():
    detector = EmergencyDetector()

    result = detector.check("FAST trong dot quy gom nhung dau hieu nao?")

    assert result.is_emergency is False


def test_guardrail_blocks_prescription_or_diagnosis_request():
    guardrails = MedicalGuardrails()

    result = guardrails.check("Hay chan doan benh va ke don thuoc cho toi")

    assert result.allowed is False
    assert "khong the chan doan" in normalize_text(result.message or "")
    assert MEDICAL_DISCLAIMER in result.message
