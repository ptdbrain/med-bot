import re

from app.evaluation.answer_eval import has_required_answer_format
from app.safety.policy import MEDICAL_DISCLAIMER
from app.schemas.answer import AnswerDraft, CitedClaim


def validate_citations(text: str, allowed: set[str]) -> list[str]:
    return sorted(set(re.findall(r"\[(\d+)\]", text)) - allowed)


def answer_errors(answer: str, allowed: set[str]) -> list[str]:
    errors = []
    if not answer.strip():
        errors.append("empty_answer")
    if not re.search(r"\[\d+\]", answer):
        errors.append("missing_citations")
    if validate_citations(answer, allowed):
        errors.append("invalid_citations")
    if not has_required_answer_format(answer):
        errors.append("invalid_format")
    if "<think>" in answer or answer.lstrip().lower().startswith(("okay, let's", "let me think", "first, i need")):
        errors.append("reasoning_content")
    return errors


def render_draft(draft: AnswerDraft, allowed: set[str]) -> str:
    used = set(re.findall(r"\[(\d+)\]", draft.short_answer + draft.explanation))
    sources = ", ".join(f"[{i}]" for i in sorted(used & allowed, key=int))
    return (
        f"1. Câu trả lời ngắn gọn\n{draft.short_answer}\n\n"
        f"2. Giải thích chi tiết\n{draft.explanation}\n\n3. Nguồn đã dùng\n{sources}\n\n"
        f"4. Cảnh báo y tế\n{MEDICAL_DISCLAIMER}"
    )


def claims_from_answer(answer: str) -> list[CitedClaim]:
    body = re.split(r"(?m)^\s*(?:\*\*)?3\.\s", answer, maxsplit=1)[0]
    claims = []
    for paragraph in body.splitlines():
        paragraph = paragraph.strip()
        if not paragraph or re.match(r"^(?:\*\*)?[12]\.\s*(?:Câu trả lời|Giải thích)", paragraph):
            continue
        ids = re.findall(r"\[(\d+)\]", paragraph)
        text = re.sub(r"\[\d+\]", "", paragraph).strip()
        if text:
            claims.append(CitedClaim(text=text, citation_ids=ids))
    return claims
