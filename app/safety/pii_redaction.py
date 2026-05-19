from __future__ import annotations

import re


EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\s.-]?){8,14}\d\b")
PATIENT_ID_PATTERN = re.compile(r"\b(?:BN|PAT|MRN)[-_]?\d{3,}\b", re.IGNORECASE)


def redact_pii(text: str) -> str:
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    return PATIENT_ID_PATTERN.sub("[REDACTED_PATIENT_ID]", text)
