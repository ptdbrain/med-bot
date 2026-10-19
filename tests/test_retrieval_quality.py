import math

import pytest

from tests.rag_helpers import make_result


def test_cross_encoder_normalizes_logits_once_and_rejects_nan():
    from app.retrieval.reranker import normalize_logit

    assert normalize_logit(0) == pytest.approx(0.5)
    assert normalize_logit(-4) < 0.02
    with pytest.raises(ValueError):
        normalize_logit(math.nan)


def test_strong_reranker_negative_cannot_be_rescued_by_keyword():
    from app.retrieval.relevance import RelevancePolicy

    result = make_result("bad", "Enzyme glossary", {"semantic_score": 0.9, "reranker_score": 0.01})
    decision = RelevancePolicy().evaluate("enzyme mechanism", result)
    assert not decision.accepted
    assert "reranker_below_threshold" in decision.reasons


def test_query_preserves_entities_and_uses_accent_fold_only_for_sparse():
    from app.retrieval.query_normalizer import normalize_query

    query = "HbA1c 6.5%, Na+ 135 mmol/L, IL-6 không tăng"
    result = normalize_query(query)
    assert result.dense_text == query
    assert "khong" in result.sparse_text


def test_bm25_updates_statistics_when_corpus_replaced():
    from app.retrieval.bm25_store import BM25Store

    store = BM25Store()
    store.replace_chunks([make_result("a", "Enzyme catalysis").chunk])
    assert store.search("Enzyme")[0].chunk.chunk_id == "a"
    store.replace_chunks([make_result("b", "Hemoglobin transport").chunk])
    assert store.search("Enzyme") == []
    assert store.search("Hemoglobin")[0].chunk.chunk_id == "b"


def test_exact_acronym_is_a_topic_anchor_but_not_a_negative_reranker_override():
    from app.retrieval.relevance import RelevancePolicy

    result = make_result("fast", "FAST includes Face, Arm, Speech and Time.")
    assert RelevancePolicy().evaluate("FAST trong dot quy gom nhung dau hieu nao?", result).accepted
    result.signals["reranker_score"] = 0.01
    assert not RelevancePolicy().evaluate("FAST signs", result).accepted
