import pytest


def test_identity_and_revision_are_separate():
    from app.ingestion.identity import document_id, revision_id

    assert document_id("study", "book.pdf") == document_id("study", "book.pdf")
    assert document_id("other", "book.pdf") != document_id("study", "book.pdf")
    assert revision_id(b"old", "v1") != revision_id(b"new", "v1")


def test_blocks_reject_invalid_page_and_confidence():
    from app.schemas.extraction import TextBlock

    with pytest.raises(ValueError):
        TextBlock("b", 0, "paragraph", "text")
    with pytest.raises(ValueError):
        TextBlock("b", 1, "paragraph", "text", confidence=2)
    assert TextBlock("b", 1, "paragraph", "text").confidence is None


def test_dataset_rejects_leakage_missing_evidence_and_unreviewed():
    from app.evaluation.dataset import EvalCase, validate_cases

    a = EvalCase(
        case_id="a",
        group_id="same",
        split="dev",
        question="Q?",
        query_language="en",
        category="exact",
        expected="answerable",
        relevant_evidence_ids=[],
        required_claims=[],
        unsupported_claims=[],
        reviewed_by=None,
    )
    b = EvalCase(**{**a.model_dump(), "case_id": "b", "split": "test"})
    errors = validate_cases([a, b], {}, require_review=True)
    assert "a:missing_evidence" in errors
    assert "same:split_leakage" in errors
    assert "a:unreviewed" in errors
