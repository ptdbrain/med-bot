from dataclasses import dataclass
from typing import Literal, Protocol

from app.rag.context_builder import PackedContext
from app.schemas.answer import CitedClaim


@dataclass(frozen=True)
class ClaimSupport:
    claim_id: str
    verdict: Literal["supported", "unsupported", "unknown"]
    evidence_ids: list[str]
    reason_code: str


class SupportVerifier(Protocol):
    def verify(self, claims: list[CitedClaim], context: PackedContext) -> list[ClaimSupport]: ...


def supported_answer(results: list[ClaimSupport], *, strict=True) -> bool:
    return bool(results) and all(r.verdict == "supported" or (not strict and r.verdict == "unknown") for r in results)


class UnavailableSupportVerifier:
    def verify(self, claims, context):
        return [
            ClaimSupport(str(i), "unknown", claim.citation_ids, "verifier_unavailable")
            for i, claim in enumerate(claims)
        ]
