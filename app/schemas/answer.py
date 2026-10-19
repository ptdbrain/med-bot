from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class CitedClaim(BaseModel):
    text: str = Field(min_length=1)
    citation_ids: list[str]


class AnswerDraft(BaseModel):
    short_answer: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    claims: list[CitedClaim]


@dataclass
class GenerationResult:
    draft: AnswerDraft | None
    finish_reason: str
    raw_text: str | None = None
    usage: dict = field(default_factory=dict)
