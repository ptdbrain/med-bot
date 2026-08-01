import tempfile
from pathlib import Path

from app.config import Settings
from app.retrieval.dense_store import SQLiteDenseVectorStore
from app.retrieval.factory import create_vector_store
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.document import Chunk, ChunkMetadata


def test_local_vector_store_returns_relevant_chunks():
    store = InMemoryVectorStore()
    store.add_chunks(
        [
            Chunk(
                chunk_id="stroke-1",
                text="Dot quy co dau hieu FAST, meo mieng, yeu tay, noi kho.",
                metadata=ChunkMetadata(
                    doc_id="stroke",
                    source_path="data/sample/stroke.md",
                    source_type="md",
                    title="Dot quy",
                    created_at="2026-05-19T00:00:00Z",
                    hash="abc",
                ),
            ),
            Chunk(
                chunk_id="lab-1",
                text="Hemoglobin la chi so xet nghiem mau.",
                metadata=ChunkMetadata(
                    doc_id="lab",
                    source_path="data/sample/lab.md",
                    source_type="md",
                    title="Xet nghiem mau",
                    created_at="2026-05-19T00:00:00Z",
                    hash="def",
                ),
            ),
        ]
    )

    results = store.search("dau hieu dot quy", top_k=1)

    assert results[0].chunk.chunk_id == "stroke-1"
    assert results[0].score > 0


class FakeMultilingualEmbedding:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @staticmethod
    def _vector(text: str) -> list[float]:
        normalized = text.lower()
        if "enzyme" in normalized or "chuyển hóa" in normalized:
            return [1.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0]


def test_sqlite_dense_store_persists_and_returns_semantic_signals():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dense.db"
        store = SQLiteDenseVectorStore(
            path=path,
            embedding_provider=FakeMultilingualEmbedding(),
            vector_size=3,
        )
        store.add_chunks(
            [
                Chunk(
                    chunk_id="enzyme-1",
                    text="Enzymes regulate metabolic reactions.",
                    metadata=ChunkMetadata(
                        doc_id="biochemistry",
                        source_path="data/raw/biochemistry.pdf",
                        source_type="pdf",
                        title="Biochemistry",
                        created_at="2026-07-31T00:00:00Z",
                        hash="enzyme-1",
                    ),
                ),
                Chunk(
                    chunk_id="stroke-1",
                    text="FAST supports stroke recognition.",
                    metadata=ChunkMetadata(
                        doc_id="stroke",
                        source_path="data/raw/stroke.md",
                        source_type="md",
                        title="Stroke",
                        created_at="2026-07-31T00:00:00Z",
                        hash="stroke-1",
                    ),
                ),
            ]
        )

        restarted = SQLiteDenseVectorStore(
            path=path,
            embedding_provider=FakeMultilingualEmbedding(),
            vector_size=3,
        )
        results = restarted.search(
            "Vai trò enzyme trong chuyển hóa",
            top_k=1,
        )

    assert results[0].chunk.chunk_id == "enzyme-1"
    assert results[0].signals["semantic_score"] == 1.0


def test_vector_factory_requires_embedding_for_sqlite_dense(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        index_path=tmp_path / "lexical.json",
        metadata_db_path=tmp_path / "metadata.db",
        dense_index_path=tmp_path / "dense.db",
        vector_store_provider="sqlite-dense",
        embedding_dimension=3,
    )

    try:
        create_vector_store(settings, None)
    except ValueError as exc:
        assert "embedding" in str(exc).lower()
    else:
        raise AssertionError("sqlite-dense must require an embedding provider")
