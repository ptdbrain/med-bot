import pytest


def test_recall_mrr_and_ndcg_use_evidence_not_document_names():
    from app.evaluation.metrics import ranking_metrics

    result = ranking_metrics(["wrong", "e1", "e2"], {"e1": 2, "e2": 1}, k=2)
    assert result["recall"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
    assert 0 < result["ndcg"] < 1


def test_duplicate_predictions_cannot_inflate_recall():
    from app.evaluation.metrics import ranking_metrics

    result = ranking_metrics(["e1", "e1", "e1"], {"e1": 1, "e2": 1}, k=3)
    assert result["recall"] == 0.5


def test_ocr_error_rate_counts_missing_diacritics():
    from app.evaluation.metrics import character_error_rate

    assert character_error_rate("không", "khong") > 0
    assert character_error_rate("enzyme", "enzyme") == 0
