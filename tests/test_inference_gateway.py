from dataclasses import replace

from fastapi.testclient import TestClient

from app.config import Settings


def test_gateway_requires_auth_and_returns_fingerprint():
    from services.inference_gateway.app import create_gateway

    class Embedding:
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    settings = replace(Settings(), http_provider_key="test-key", embedding_dimension=2)
    client = TestClient(create_gateway(settings, embedding=Embedding()))
    assert client.post("/v1/embeddings", json={"texts": ["Q"], "purpose": "query"}).status_code == 401
    response = client.post(
        "/v1/embeddings", json={"texts": ["Q"], "purpose": "query"}, headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200
    assert response.json()["vectors"] == [[1.0, 0.0]]
    assert response.json()["model_fingerprint"]
