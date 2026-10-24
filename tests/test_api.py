from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.runtime.doctor import RuntimeCheck


def _test_settings(tmp_path):
    return Settings(
        data_dir=tmp_path,
        index_path=tmp_path / "indexes" / "vector_store.json",
        metadata_db_path=tmp_path / "indexes" / "metadata.db",
    )


def test_chat_api_returns_answer_and_sources_after_ingest(tmp_path):
    sample = tmp_path / "stroke.md"
    sample.write_text(
        "# Dot quy\n\nFAST gom meo mieng, yeu tay, noi kho va can goi cap cuu som.",
        encoding="utf-8",
    )
    client = TestClient(create_app(_test_settings(tmp_path)))

    ingest_response = client.post(
        "/ingest/file",
        files={"file": ("stroke.md", sample.read_bytes(), "text/markdown")},
    )
    assert ingest_response.status_code == 200

    chat_response = client.post(
        "/chat",
        json={"question": "FAST trong dot quy gom nhung dau hieu nao?", "rag_mode": "naive", "top_k": 3},
    )

    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert payload["answer"]
    assert payload["sources"]
    assert payload["mode_used"] == "naive"
    assert "Thông tin chỉ phục vụ học tập/tham khảo" in payload["answer"]


def test_chat_api_reports_insufficient_context_when_store_empty(tmp_path):
    client = TestClient(create_app(_test_settings(tmp_path)))

    response = client.post(
        "/chat",
        json={"question": "Noi dung nay khong co trong tai lieu nao?", "rag_mode": "naive", "top_k": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["sources"] == []
    assert "Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp." in payload["answer"]


def test_retrieve_api_rejects_unknown_rag_mode(tmp_path):
    client = TestClient(create_app(_test_settings(tmp_path)))

    response = client.get(
        "/retrieve",
        params={
            "query": "FAST",
            "rag_mode": "unknown",
            "top_k": 3,
        },
    )

    assert response.status_code == 400
    assert "RAG mode" in response.json()["detail"]


def test_health_reports_active_rag_providers_and_vector_count(tmp_path):
    client = TestClient(create_app(_test_settings(tmp_path)))

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["chunks_indexed"] == 0
    assert payload["vectors_indexed"] == 0
    assert payload["embedding_provider"] == "lexical"
    assert payload["vector_store_provider"] == "memory"
    assert payload["reranker_provider"] == "lexical"


def test_ready_returns_503_when_required_runtime_component_is_missing(tmp_path):
    client = TestClient(create_app(_test_settings(tmp_path)))
    with patch("app.api.health.check_runtime", return_value=[RuntimeCheck("qdrant", "missing", "offline")]):
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["checks"][0]["name"] == "qdrant"
