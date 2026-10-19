from app.llm.base import LLMProvider
from app.providers.http_transport import HTTPTransport
from app.rag.answer_validation import render_draft
from app.rag.context_builder import context_payload
from app.rag.support_verifier import ClaimSupport
from app.schemas.answer import AnswerDraft, GenerationResult


class HTTPLLM(LLMProvider):
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport or HTTPTransport(settings)

    def generate_result(self, question, contexts):
        body = self.transport.post(
            "/v1/generate",
            {
                "question": question,
                "sources": context_payload(contexts),
                "schema": AnswerDraft.model_json_schema(),
                "max_output_tokens": self.settings.context_output_reserve,
            },
        )
        draft = AnswerDraft.model_validate(body["draft"]) if body.get("draft") else None
        return GenerationResult(
            draft, body.get("finish_reason", "unknown"), body.get("raw_text"), body.get("usage", {})
        )

    def generate(self, question, contexts):
        result = self.generate_result(question, contexts)
        if result.finish_reason != "stop":
            raise RuntimeError("Remote generation did not complete")
        return (
            render_draft(result.draft, {str(i) for i in range(1, len(contexts) + 1)})
            if result.draft
            else result.raw_text or ""
        )


class HTTPSupportVerifier:
    def __init__(self, settings):
        self.transport = HTTPTransport(settings)

    def verify(self, claims, context):
        response = self.transport.post(
            "/v1/verify", {"claims": [c.model_dump() for c in claims], "source_map": context.source_map}
        )
        values = [ClaimSupport(**row) for row in response["results"]]
        if len(values) != len(claims) or [v.claim_id for v in values] != [str(i) for i in range(len(claims))]:
            raise RuntimeError("Invalid remote verifier response")
        return values
