import pytest


def test_retry_policy_excludes_auth_errors():
    from app.providers.http_transport import is_retryable_status

    assert is_retryable_status(429)
    assert is_retryable_status(503)
    assert not is_retryable_status(401)


def test_embedding_response_must_match_shape_and_fingerprint():
    from app.embeddings.http import validate_vectors

    with pytest.raises(ValueError, match="fingerprint"):
        validate_vectors({"model_fingerprint": "wrong", "vectors": [[1, 0]]}, 1, 2, "expected")
    with pytest.raises(ValueError):
        validate_vectors({"model_fingerprint": "expected", "vectors": [[float("nan"), 0]]}, 1, 2, "expected")
    assert validate_vectors({"model_fingerprint": "expected", "vectors": [[1, 0]]}, 1, 2, "expected") == [[1.0, 0.0]]
