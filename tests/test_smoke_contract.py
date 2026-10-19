import pytest

from scripts.smoke_rag import check_answer_payload


def test_smoke_requires_nonempty_answer():
    errors = check_answer_payload({"answer": "", "sources": [], "mode_used": "rerank"})
    assert "empty_answer" in errors


def test_smoke_requires_a_citation_when_sources_exist():
    errors = check_answer_payload({"answer": "An answer.", "sources": [{"chunk_id": "chunk-1"}]})
    assert "missing_citation" in errors


@pytest.mark.parametrize(
    ("answer", "sources", "expected"),
    [
        ("Reasoning: let me think", [], "reasoning_only"),
        ("Enzyme X [99]", [], "missing_evidence"),
        ("An answer [99]", [{"chunk_id": "chunk-1"}], "invalid_citation_id"),
        ("An answer [0]", [{"chunk_id": "chunk-1"}], "invalid_citation_id"),
    ],
)
def test_smoke_rejects_unusable_answer(answer, sources, expected):
    errors = check_answer_payload({"answer": answer, "sources": sources, "mode_used": "rerank"})
    assert expected in errors
