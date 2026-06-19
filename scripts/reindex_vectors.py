from __future__ import annotations

import argparse

from app.config import get_settings
from app.embeddings.factory import create_embedding_provider
from app.retrieval.factory import create_vector_store
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.vector_store import VectorStore
from app.schemas.document import Chunk


def reindex_chunks(
    chunks: list[Chunk],
    vector_store: VectorStore,
    *,
    clear: bool = False,
) -> int:
    if clear:
        clear_store = getattr(vector_store, "clear", None)
        if not callable(clear_store):
            raise RuntimeError("Vector store hiện tại không hỗ trợ xóa collection.")
        clear_store()
    vector_store.add_chunks(chunks)
    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upsert các chunk trong SQLite sang vector store đã cấu hình."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa vector store trước khi lập chỉ mục lại.",
    )
    args = parser.parse_args()

    settings = get_settings()
    metadata_store = SQLiteMetadataStore(settings.metadata_db_path)
    chunks = metadata_store.list_chunks()
    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings, embedding_provider)
    count = reindex_chunks(chunks, vector_store, clear=args.clear)
    print(
        f"reindexed {count} chunks into "
        f"{settings.vector_store_provider}:{settings.qdrant_collection}"
    )


if __name__ == "__main__":
    main()
