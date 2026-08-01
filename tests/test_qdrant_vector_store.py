from __future__ import annotations

import unittest

from app.retrieval.vector_store import QdrantVectorStore
from app.schemas.document import Chunk, ChunkMetadata


class FakeEmbedding:
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [0.4, 0.5, 0.6]


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.text = str(self.payload)

    def json(self) -> dict:
        return self.payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def _request(self, method: str, url: str, **kwargs) -> FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)

    def get(self, url: str, **kwargs) -> FakeResponse:
        return self._request("GET", url, **kwargs)

    def put(self, url: str, **kwargs) -> FakeResponse:
        return self._request("PUT", url, **kwargs)

    def post(self, url: str, **kwargs) -> FakeResponse:
        return self._request("POST", url, **kwargs)

    def delete(self, url: str, **kwargs) -> FakeResponse:
        return self._request("DELETE", url, **kwargs)


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="stroke-fast-1",
        text="FAST gồm méo miệng, yếu tay và nói khó.",
        metadata=ChunkMetadata(
            doc_id="stroke",
            source_path="stroke.pdf",
            source_type="pdf",
            title="Hướng dẫn đột quỵ",
            created_at="2026-06-11T00:00:00Z",
            hash="abc",
            page_number=3,
            section_title="FAST",
        ),
    )


class QdrantVectorStoreTests(unittest.TestCase):
    def test_missing_collection_is_created_with_cosine_vectors(self) -> None:
        session = FakeSession(
            [
                FakeResponse(404),
                FakeResponse(200, {"result": True}),
            ]
        )

        QdrantVectorStore(
            url="http://qdrant.test:6333/",
            collection="medical",
            embedding_provider=FakeEmbedding(),
            vector_size=3,
            session=session,
        )

        self.assertEqual(session.calls[0]["method"], "GET")
        self.assertEqual(
            session.calls[0]["url"],
            "http://qdrant.test:6333/collections/medical",
        )
        self.assertEqual(session.calls[1]["method"], "PUT")
        self.assertEqual(
            session.calls[1]["json"],
            {"vectors": {"size": 3, "distance": "Cosine"}},
        )

    def test_add_chunks_embeds_and_upserts_complete_payload(self) -> None:
        embedding = FakeEmbedding()
        session = FakeSession(
            [
                FakeResponse(200, {"result": {}}),
                FakeResponse(200, {"result": {"status": "completed"}}),
            ]
        )
        store = QdrantVectorStore(
            url="http://qdrant.test:6333",
            collection="medical",
            embedding_provider=embedding,
            vector_size=3,
            session=session,
        )

        store.add_chunks([_chunk()])

        self.assertEqual(
            embedding.document_calls,
            [["Hướng dẫn đột quỵ\nFAST\nFAST gồm méo miệng, yếu tay và nói khó."]],
        )
        upsert = session.calls[1]
        self.assertEqual(upsert["method"], "PUT")
        self.assertIn("/points", upsert["url"])
        point = upsert["json"]["points"][0]
        self.assertEqual(point["vector"], [0.1, 0.2, 0.3])
        self.assertEqual(point["payload"]["chunk"]["chunk_id"], "stroke-fast-1")
        self.assertEqual(
            point["payload"]["chunk"]["metadata"]["section_title"],
            "FAST",
        )
        self.assertEqual(len(point["id"]), 36)

    def test_search_reconstructs_ranked_search_results(self) -> None:
        chunk = _chunk()
        embedding = FakeEmbedding()
        session = FakeSession(
            [
                FakeResponse(200, {"result": {}}),
                FakeResponse(
                    200,
                    {
                        "result": [
                            {
                                "id": "point-1",
                                "score": 0.87,
                                "payload": {"chunk": chunk.to_dict()},
                            }
                        ]
                    },
                ),
            ]
        )
        store = QdrantVectorStore(
            url="http://qdrant.test:6333",
            collection="medical",
            embedding_provider=embedding,
            vector_size=3,
            session=session,
        )

        results = store.search("dấu hiệu FAST", top_k=4, min_score=0.25)

        self.assertEqual(embedding.query_calls, ["dấu hiệu FAST"])
        search = session.calls[1]
        self.assertEqual(search["method"], "POST")
        self.assertTrue(search["url"].endswith("/points/search"))
        self.assertEqual(search["json"]["vector"], [0.4, 0.5, 0.6])
        self.assertEqual(search["json"]["limit"], 4)
        self.assertEqual(search["json"]["score_threshold"], 0.25)
        self.assertEqual(results[0].chunk.chunk_id, "stroke-fast-1")
        self.assertEqual(results[0].chunk.metadata.page_number, 3)
        self.assertEqual(results[0].score, 0.87)

    def test_delete_chunks_removes_deterministic_point_ids(self) -> None:
        session = FakeSession(
            [
                FakeResponse(200, {"result": {}}),
                FakeResponse(200, {"result": {"status": "completed"}}),
            ]
        )
        store = QdrantVectorStore(
            url="http://qdrant.test:6333",
            collection="medical",
            embedding_provider=FakeEmbedding(),
            vector_size=3,
            session=session,
        )

        store.delete_chunks(["stroke-fast-1"])

        delete_call = session.calls[1]
        self.assertEqual(delete_call["method"], "POST")
        self.assertTrue(delete_call["url"].endswith("/points/delete"))
        self.assertTrue(delete_call["params"]["wait"])
        self.assertEqual(len(delete_call["json"]["points"][0]), 36)

    def test_qdrant_http_failure_mentions_service_url(self) -> None:
        session = FakeSession([FakeResponse(503, {"status": "unavailable"})])

        with self.assertRaisesRegex(RuntimeError, "qdrant.test:6333"):
            QdrantVectorStore(
                url="http://qdrant.test:6333",
                collection="medical",
                embedding_provider=FakeEmbedding(),
                vector_size=3,
                session=session,
            )


if __name__ == "__main__":
    unittest.main()
