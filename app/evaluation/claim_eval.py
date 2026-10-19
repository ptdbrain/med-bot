from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE

# Common English & Vietnamese stopwords to filter when extracting claim keywords
STOPWORDS = {
    # English
    "a", "an", "the", "and", "or", "is", "are", "was", "were", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "that", "this", "it", "its",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    # Vietnamese
    "và", "hoặc", "là", "thì", "mà", "ở", "tại", "cho", "của", "với", "bởi",
    "từ", "như", "được", "bị", "các", "những", "một", "này", "đó", "có", "không",
    "cũng", "đã", "đang", "sẽ", "trong", "vào", "ra", "lên", "xuống", "đến", "dùng",
    "khi", "lại", "nào", "gì", "theo",
}


def _clean_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).casefold()
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def verify_claim_support(claim: str, answer: str, threshold: float = 0.45) -> tuple[bool, float]:
    """Verify if a required factual claim is supported/entailed in the answer.

    Uses keyword constituent coverage and sentence-level matching.
    Returns (is_supported, match_score).
    """
    claim_norm = unicodedata.normalize("NFC", claim).strip().casefold()
    answer_norm = unicodedata.normalize("NFC", answer).strip().casefold()

    if not claim_norm:
        return True, 1.0

    # 1. Exact or normalized phrase match
    if claim_norm in answer_norm:
        return True, 1.0

    claim_tokens = _clean_tokens(claim)
    if not claim_tokens:
        return True, 1.0

    answer_tokens_set = set(_clean_tokens(answer))
    matched_tokens = sum(1 for t in claim_tokens if t in answer_tokens_set)
    global_score = matched_tokens / len(claim_tokens)

    # 2. Check sentence-level alignment for multi-sentence answers
    sentences = re.split(r"[.\n;!?]+", answer_norm)
    best_sent_score = 0.0
    for sent in sentences:
        sent_tokens = set(_clean_tokens(sent))
        if not sent_tokens:
            continue
        sent_matches = sum(1 for t in claim_tokens if t in sent_tokens)
        sent_score = sent_matches / len(claim_tokens)
        best_sent_score = max(best_sent_score, sent_score)

    effective_score = max(global_score, best_sent_score)
    is_supported = effective_score >= threshold
    return is_supported, effective_score


def detect_unsupported_claim(unsupported_claim: str, answer: str, threshold: float = 0.85) -> tuple[bool, float]:
    """Detect if an unsupported/hallucinated assertion is present in the answer.

    Requires phrase match or high token co-occurrence WITHIN THE SAME SENTENCE
    to prevent false positives across unrelated sentences.
    Returns (is_detected, detection_score).
    """
    claim_norm = unicodedata.normalize("NFC", unsupported_claim).strip().casefold()
    answer_norm = unicodedata.normalize("NFC", answer).strip().casefold()

    if not claim_norm:
        return False, 0.0

    # Contiguous phrase match
    if claim_norm in answer_norm:
        return True, 1.0

    claim_tokens = _clean_tokens(unsupported_claim)
    if len(claim_tokens) < 2:
        return False, 0.0

    # Check sentence by sentence so tokens in paragraph 1 don't combine with paragraph 2
    sentences = re.split(r"[.\n;!?]+", answer_norm)
    best_sent_score = 0.0
    for sent in sentences:
        sent_tokens = set(_clean_tokens(sent))
        if not sent_tokens:
            continue
        matched = sum(1 for t in claim_tokens if t in sent_tokens)
        sent_score = matched / len(claim_tokens)
        best_sent_score = max(best_sent_score, sent_score)

    is_detected = best_sent_score >= threshold
    return is_detected, best_sent_score


def evaluate_claims(
    answer: str,
    required_claims: list[str],
    unsupported_claims: list[str],
    expected: str = "answerable",
) -> dict[str, Any]:
    """Evaluate claim-level accuracy, hallucination detection, and faithfulness for an answer."""
    if expected == "insufficient_context":
        refused = (
            INSUFFICIENT_CONTEXT_MESSAGE.casefold() in answer.casefold()
            or "không đủ" in answer.casefold()
            or "không có thông tin" in answer.casefold()
            or "insufficient" in answer.casefold()
        )
        return {
            "expected": expected,
            "required_claims_total": 0,
            "required_claims_satisfied": 0,
            "claim_recall": 1.0 if refused else 0.0,
            "unsupported_claims_total": len(unsupported_claims),
            "unsupported_claims_detected": 0 if refused else len(unsupported_claims),
            "unsupported_claim_rate": 0.0 if refused else 1.0,
            "faithfulness_score": 1.0 if refused else 0.0,
            "passed": refused,
            "claim_details": [],
            "unsupported_details": [],
        }

    # Evaluate required claims
    claim_details = []
    satisfied_count = 0
    for claim in required_claims:
        supported, score = verify_claim_support(claim, answer)
        if supported:
            satisfied_count += 1
        claim_details.append(
            {
                "claim": claim,
                "supported": supported,
                "match_score": round(score, 3),
            }
        )

    claim_recall = satisfied_count / len(required_claims) if required_claims else 1.0

    # Evaluate unsupported claims
    unsupported_details = []
    detected_unsupported = 0
    for claim in unsupported_claims:
        detected, score = detect_unsupported_claim(claim, answer)
        if detected:
            detected_unsupported += 1
        unsupported_details.append(
            {
                "claim": claim,
                "detected": detected,
                "detection_score": round(score, 3),
            }
        )

    unsupported_rate = (
        detected_unsupported / len(unsupported_claims) if unsupported_claims else 0.0
    )

    # Faithfulness score: rewards recall, heavily penalizes hallucinations
    penalty = min(1.0, 0.5 * detected_unsupported)
    faithfulness_score = max(0.0, claim_recall * (1.0 - penalty))

    passed = (claim_recall >= 0.7) and (detected_unsupported == 0)

    return {
        "expected": expected,
        "required_claims_total": len(required_claims),
        "required_claims_satisfied": satisfied_count,
        "claim_recall": round(claim_recall, 4),
        "unsupported_claims_total": len(unsupported_claims),
        "unsupported_claims_detected": detected_unsupported,
        "unsupported_claim_rate": round(unsupported_rate, 4),
        "faithfulness_score": round(faithfulness_score, 4),
        "passed": passed,
        "claim_details": claim_details,
        "unsupported_details": unsupported_details,
    }
