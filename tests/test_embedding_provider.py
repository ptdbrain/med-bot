from __future__ import annotations

import unittest

from app.config import Settings
from app.embeddings.factory import create_embedding_provider
from app.embeddings.sentence_transformer import SentenceTransformerEmbedding


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(self, texts: list[str], **kwargs):
        self.calls.append({"texts": texts, **kwargs})
        return [[float(index), 0.5] for index, _ in enumerate(texts, start=1)]


class EmbeddingProviderTests(unittest.TestCase):
    def test_sentence_transformer_normalizes_document_embeddings(self) -> None:
        model = FakeModel()
        provider = SentenceTransformerEmbedding(
            model_name="BAAI/bge-m3",
            device="cpu",
            batch_size=4,
            model=model,
        )

        vectors = provider.embed_documents(["chunk one", "chunk two"])

        self.assertEqual(vectors, [[1.0, 0.5], [2.0, 0.5]])
        self.assertEqual(model.calls[0]["texts"], ["chunk one", "chunk two"])
        self.assertEqual(model.calls[0]["batch_size"], 4)
        self.assertTrue(model.calls[0]["normalize_embeddings"])
        self.assertFalse(model.calls[0]["show_progress_bar"])

    def test_query_embedding_returns_one_vector(self) -> None:
        provider = SentenceTransformerEmbedding(
            model_name="BAAI/bge-m3",
            model=FakeModel(),
        )

        vector = provider.embed_query("FAST là gì?")

        self.assertEqual(vector, [1.0, 0.5])

    def test_lexical_provider_does_not_construct_dense_model(self) -> None:
        provider = create_embedding_provider(
            Settings(embedding_provider="lexical")
        )

        self.assertIsNone(provider)

    def test_factory_builds_sentence_transformer_provider(self) -> None:
        provider = create_embedding_provider(
            Settings(
                embedding_provider="sentence-transformers",
                embedding_model="BAAI/bge-m3",
                embedding_device="cpu",
                embedding_batch_size=6,
            )
        )

        self.assertIsInstance(provider, SentenceTransformerEmbedding)
        self.assertEqual(provider.model_name, "BAAI/bge-m3")
        self.assertEqual(provider.device, "cpu")
        self.assertEqual(provider.batch_size, 6)
        self.assertIsNone(provider._model)

    def test_factory_rejects_unknown_embedding_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "EMBEDDING_PROVIDER"):
            create_embedding_provider(
                Settings(embedding_provider="unknown")
            )


if __name__ == "__main__":
    unittest.main()
