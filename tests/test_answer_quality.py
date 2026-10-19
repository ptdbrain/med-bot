from app.safety.medical_guardrails import MedicalGuardrails
from tests.rag_helpers import make_result


def test_context_budget_omits_oversized_chunks_and_preserves_numbering():
    from app.rag.context_builder import ContextBuilder

    class Counter:
        def count(self, text):
            return len(text)

    packed = ContextBuilder(Counter()).pack(
        "Q?", [make_result("long", "x" * 5000), make_result("short", "Evidence")], input_budget=400
    )
    assert packed.token_count <= 400
    assert packed.source_map["1"]["chunk_id"] == "short"
    assert "long" in packed.omitted_ids


def test_citation_ids_must_exist():
    from app.rag.answer_validation import validate_citations

    assert validate_citations("Fact [1], invention [99].", {"1"}) == ["99"]


def test_invalid_answer_is_repaired_once_then_refused():
    from app.rag.answer_service import AnswerService
    from app.rag.context_builder import ContextBuilder

    class BrokenLLM:
        calls = 0

        def generate(self, question, contexts):
            self.calls += 1
            return "Fact [99]"

    llm = BrokenLLM()
    packed = ContextBuilder().pack("Q?", [make_result("a", "Evidence")], input_budget=2000)
    output = AnswerService(llm, MedicalGuardrails()).answer("Q?", packed)
    assert llm.calls == 2
    assert output.results == []
    assert "không" in output.answer.lower()


def test_unknown_support_is_not_accepted_in_strict_mode():
    from app.rag.support_verifier import ClaimSupport, supported_answer

    assert not supported_answer([ClaimSupport("c1", "unknown", ["1"], "timeout")], strict=True)
