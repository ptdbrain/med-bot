from __future__ import annotations

import re
from dataclasses import asdict

from app.llm.mock_llm import INSUFFICIENT_CONTEXT_MESSAGE
from app.rag.answer_validation import answer_errors, claims_from_answer, render_draft
from app.rag.naive_rag import RagOutput
from app.rag.support_verifier import UnavailableSupportVerifier, supported_answer


class AnswerService:
    def __init__(self, llm, guardrails, *, strict=False, verifier=None):
        self.llm, self.guardrails, self.strict = llm, guardrails, strict
        self.verifier = verifier or UnavailableSupportVerifier()

    def answer(self, question, packed_context):
        safety = self.guardrails.check(question)
        if not safety.allowed:
            return RagOutput(safety.message or "", [], safety.safety_warning)
        if not packed_context.items:
            return RagOutput(self.guardrails.with_disclaimer(INSUFFICIENT_CONTEXT_MESSAGE), [])
        allowed = set(packed_context.source_map)
        prompt, errors, supports = question, [], []
        for attempt in range(2):
            if hasattr(self.llm, "generate_result"):
                generated = self.llm.generate_result(prompt, packed_context.items)
                text = render_draft(generated.draft, allowed) if generated.draft else generated.raw_text or ""
                errors = ["truncated_generation"] if generated.finish_reason == "length" else []
            else:
                text = self.llm.generate(prompt, packed_context.items)
                errors = []
            errors += answer_errors(text, allowed)
            if not errors and self.strict:
                claims = claims_from_answer(text)
                supports = self.verifier.verify(claims, packed_context)
                if len(supports) != len(claims) or not supported_answer(supports):
                    if any(s.reason_code == "verifier_unavailable" for s in supports):
                        raise RuntimeError("Answer support verifier is unavailable")
                    errors.append("unsupported_claims")
            if not errors:
                final = self.guardrails.with_disclaimer(text)
                used = set(re.findall(r"\[(\d+)\]", final))
                return RagOutput(
                    final,
                    packed_context.items,
                    validation={
                        "level": "strict" if self.strict else "basic",
                        "cited_ids": sorted(used),
                        "attempts": attempt + 1,
                        "support": [asdict(s) for s in supports],
                        "source_map": packed_context.source_map,
                    },
                )
            prompt = (
                question + "\nCorrect the following validation errors using only the same sources: " + ", ".join(errors)
            )
        return RagOutput(
            self.guardrails.with_disclaimer(INSUFFICIENT_CONTEXT_MESSAGE),
            [],
            validation={"level": "rejected", "errors": errors, "attempts": 2},
        )
