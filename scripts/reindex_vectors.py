from __future__ import annotations

import argparse
from collections.abc import Callable

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
    batch_size: int = 64,
    resume: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if clear:
        clear_store = getattr(vector_store, "clear", None)
        if not callable(clear_store):
            raise RuntimeError("Vector store hiện tại không hỗ trợ xóa collection.")
        clear_store()

    pending = chunks
    if resume and not clear:
        get_existing_ids = getattr(
            vector_store,
            "existing_chunk_ids",
            None,
        )
        if not callable(get_existing_ids):
            raise RuntimeError("Vector store hiện tại không hỗ trợ resume.")
        existing_ids = get_existing_ids()
        pending = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]

    total = len(pending)
    for start in range(0, total, batch_size):
        batch = pending[start : start + batch_size]
        vector_store.add_chunks(batch)
        done = min(start + len(batch), total)
        if progress_callback is not None:
            progress_callback(done, total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert các chunk trong SQLite sang vector store đã cấu hình.")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Xóa vector store trước khi lập chỉ mục lại.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bỏ qua chunk đã có trong vector store.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Số chunk được upsert trong mỗi batch.",
    )
    args = parser.parse_args()

    settings = get_settings()
    metadata_store = SQLiteMetadataStore(settings.metadata_db_path)
    chunks = metadata_store.list_chunks()
    embedding_provider = create_embedding_provider(settings)
    vector_store = create_vector_store(settings, embedding_provider)
    count = reindex_chunks(
        chunks,
        vector_store,
        clear=args.clear,
        batch_size=args.batch_size,
        resume=args.resume,
        progress_callback=lambda done, total: print(
            f"reindexed {done}/{total} chunks",
            flush=True,
        ),
    )
    print(f"reindexed {count} chunks into " f"{settings.vector_store_provider}:{settings.qdrant_collection}")


if __name__ == "__main__":
    main()
