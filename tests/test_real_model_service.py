from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.llm.ollama_llm import OllamaLLM
from app.rag.service import RAGService
from app.retrieval.factory import create_vector_store
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.document import Chunk, ChunkMetadata
from scripts.reindex_vectors import reindex_chunks


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="persisted-1",
        text="Hemoglobin vận chuyển oxy trong hồng cầu.",
        metadata=ChunkMetadata(
            doc_id="blood",
            source_path="blood.md",
            source_type="md",
            title="Xét nghiệm máu",
            created_at="2026-06-11T00:00:00Z",
            hash="hash-1",
            section_title="Hemoglobin",
        ),
    )


class FakeEmbedding(EmbeddingProvider):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class FakeVectorStore:
    def __init__(self) -> None:
        self.added: list[Chunk] = []
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.added.extend(chunks)


class RealModelServiceTests(unittest.TestCase):
    def test_metadata_store_lists_persisted_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteMetadataStore(Path(tmp) / "metadata.db")
            store.upsert_chunks([_chunk()])

            chunks = store.list_chunks()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "persisted-1")
        self.assertEqual(chunks[0].metadata.section_title, "Hemoglobin")

    def test_default_vector_store_factory_preserves_memory_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = create_vector_store(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                ),
                embedding_provider=None,
            )

        self.assertIsInstance(store, InMemoryVectorStore)

    def test_qdrant_requires_dense_embedding_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "Qdrant.*embedding"):
            create_vector_store(
                Settings(vector_store_provider="qdrant"),
                embedding_provider=None,
            )

    def test_service_constructs_configured_ollama_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    llm_provider="ollama",
                    ollama_base_url="http://ollama.test:11434",
                    ollama_model="qwen3:4b",
                    data_dir=root,
                    index_path=root / "indexes" / "vectors.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                )
            )

        self.assertIsInstance(service.llm, OllamaLLM)
        self.assertEqual(service.llm.base_url, "http://ollama.test:11434")

    def test_service_rebuilds_bm25_from_sqlite_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = Settings(
                data_dir=root,
                index_path=root / "indexes" / "vectors.json",
                metadata_db_path=root / "indexes" / "metadata.db",
            )
            metadata = SQLiteMetadataStore(settings.metadata_db_path)
            metadata.upsert_chunks([_chunk()])

            service = RAGService(settings)
            results = service.bm25_store.search("hemoglobin oxy", top_k=1)

            self.assertEqual(results[0].chunk.chunk_id, "persisted-1")
            self.assertEqual(service.indexed_chunk_count, 1)

    def test_reindex_chunks_can_clear_and_upsert_persisted_metadata(self) -> None:
        vector_store = FakeVectorStore()

        count = reindex_chunks([_chunk()], vector_store, clear=True)

        self.assertEqual(count, 1)
        self.assertEqual(vector_store.clear_calls, 1)
        self.assertEqual(vector_store.added[0].chunk_id, "persisted-1")


if __name__ == "__main__":
    unittest.main()
