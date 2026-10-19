from __future__ import annotations

import re
import unicodedata
from typing import Any

# Vietnamese vowels with diacritics / tone marks
VIETNAMESE_DIACRITIC_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)


def edit_distance(seq1: list[Any] | str, seq2: list[Any] | str) -> int:
    """Compute standard Levenshtein distance between two sequences."""
    row = list(range(len(seq2) + 1))
    for i, a in enumerate(seq1, 1):
        new = [i]
        for j, b in enumerate(seq2, 1):
            cost = 0 if a == b else 1
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + cost))
        row = new
    return row[-1]


def normalize_text_spacing(text: str) -> str:
    """Normalize whitespace and strip punctuation spaces."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER = Levenshtein / len(reference))."""
    ref_norm = unicodedata.normalize("NFC", reference.strip())
    hyp_norm = unicodedata.normalize("NFC", hypothesis.strip())
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    distance = edit_distance(ref_norm, hyp_norm)
    return distance / len(ref_norm)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER = Levenshtein on words / len(reference_words))."""
    ref_words = normalize_text_spacing(reference).split()
    hyp_words = normalize_text_spacing(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    distance = edit_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def strip_vietnamese_diacritics(text: str) -> str:
    """Strip tone and diacritic marks from Vietnamese text."""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    # Handle 'đ' / 'Đ'
    stripped = stripped.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", stripped)


def diacritic_metrics(reference: str, hypothesis: str) -> dict[str, float]:
    """Calculate preservation of Vietnamese diacritic marks between reference and hypothesis.
    
    Returns:
        - diacritic_char_count: total characters with diacritics in reference
        - diacritic_preservation_rate: fraction of diacritic characters correctly preserved [0.0 - 1.0]
        - diacritic_error_rate: fraction of diacritic characters missed or corrupted
    """
    ref_norm = unicodedata.normalize("NFC", reference)
    hyp_norm = unicodedata.normalize("NFC", hypothesis)

    ref_diacritics = [c for c in ref_norm if c in VIETNAMESE_DIACRITIC_CHARS]
    if not ref_diacritics:
        return {
            "diacritic_char_count": 0,
            "diacritic_preservation_rate": 1.0,
            "diacritic_error_rate": 0.0,
        }

    # Count how many diacritic chars from reference appear intact in hypothesis (multiset matching)
    hyp_counts: dict[str, int] = {}
    for c in hyp_norm:
        if c in VIETNAMESE_DIACRITIC_CHARS:
            hyp_counts[c] = hyp_counts.get(c, 0) + 1

    preserved = 0
    for c in ref_diacritics:
        if hyp_counts.get(c, 0) > 0:
            preserved += 1
            hyp_counts[c] -= 1

    preservation_rate = preserved / len(ref_diacritics)
    return {
        "diacritic_char_count": len(ref_diacritics),
        "diacritic_preservation_rate": preservation_rate,
        "diacritic_error_rate": 1.0 - preservation_rate,
    }


def quote_containment(
    quote: str,
    text: str,
    *,
    fuzzy_threshold: float = 0.85,
) -> dict[str, Any]:
    """Check whether a reference evidence quote is contained in the extracted hypothesis text.
    
    Evaluates:
    - exact_contained: raw quote substring match
    - normalized_contained: casefolded and whitespace-normalized substring match
    - diacritic_insensitive_contained: match ignoring tone marks
    - token_recall: proportion of reference quote words found in hypothesis
    - passed: overall boolean indicator meeting the fuzzy threshold
    """
    q_raw = quote.strip()
    t_raw = text.strip()

    if not q_raw:
        return {
            "exact_contained": True,
            "normalized_contained": True,
            "diacritic_insensitive_contained": True,
            "token_recall": 1.0,
            "passed": True,
        }

    exact_contained = q_raw in t_raw
    q_norm = normalize_text_spacing(q_raw).casefold()
    t_norm = normalize_text_spacing(t_raw).casefold()
    normalized_contained = q_norm in t_norm

    q_stripped = strip_vietnamese_diacritics(q_norm)
    t_stripped = strip_vietnamese_diacritics(t_norm)
    diacritic_insensitive_contained = q_stripped in t_stripped

    # Token overlap recall
    q_words = re.findall(r"\w+", q_norm)
    t_words_set = set(re.findall(r"\w+", t_norm))
    if q_words:
        matched_words = sum(1 for w in q_words if w in t_words_set)
        token_recall = matched_words / len(q_words)
    else:
        token_recall = 1.0

    passed = exact_contained or normalized_contained or (token_recall >= fuzzy_threshold)

    return {
        "exact_contained": exact_contained,
        "normalized_contained": normalized_contained,
        "diacritic_insensitive_contained": diacritic_insensitive_contained,
        "token_recall": token_recall,
        "passed": passed,
    }
