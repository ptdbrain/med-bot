import math

from app.embeddings.base import EmbeddingProvider
from app.providers.http_transport import HTTPTransport


def validate_vectors(payload: dict, count: int, dimension: int, fingerprint: str) -> list[list[float]]:
    if payload.get("model_fingerprint") != fingerprint:
        raise ValueError("Remote embedding fingerprint mismatch")
    vectors = payload.get("vectors", [])
    if len(vectors) != count:
        raise ValueError("Remote embedding count mismatch")
    output = []
    for vector in vectors:
        if len(vector) != dimension or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            raise ValueError("Invalid remote embedding vector")
        if sum(float(v) * float(v) for v in vector) == 0:
            raise ValueError("Zero embedding vector")
        output.append([float(v) for v in vector])
    return output


class HTTPEmbedding(EmbeddingProvider):
    def __init__(self, settings, transport=None):
        if not settings.http_embedding_fingerprint:
            raise ValueError("HTTP_EMBEDDING_FINGERPRINT must be configured")
        self.settings, self.transport = settings, transport or HTTPTransport(settings)

    def _embed(self, texts, purpose):
        if not texts:
            return []
        payload = self.transport.post("/v1/embeddings", {"texts": texts, "purpose": purpose})
        return validate_vectors(
            payload, len(texts), self.settings.embedding_dimension, self.settings.http_embedding_fingerprint
        )

    def embed_documents(self, texts):
        return self._embed(texts, "document")

    def embed_query(self, text):
        return self._embed([text], "query")[0]
