from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_name: str = "MedTutor Multimodal RAG Copilot"
    llm_provider: str = "mock"
    openai_api_key: str = ""
    embedding_provider: str = "local"
    qdrant_url: str = "http://qdrant:6333"
    database_url: str = "sqlite:///data/indexes/metadata.db"
    use_mock_llm: bool = True
    use_local_embeddings: bool = True
    max_upload_bytes: int = 15 * 1024 * 1024
    data_dir: Path = Path("data")
    index_path: Path = Path("data/indexes/vector_store.json")
    metadata_db_path: Path = Path("data/indexes/metadata.db")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local"),
        qdrant_url=os.getenv("QDRANT_URL", "http://qdrant:6333"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/indexes/metadata.db"),
        use_mock_llm=_bool_env("USE_MOCK_LLM", True),
        use_local_embeddings=_bool_env("USE_LOCAL_EMBEDDINGS", True),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
        data_dir=data_dir,
        index_path=data_dir / "indexes" / "vector_store.json",
        metadata_db_path=data_dir / "indexes" / "metadata.db",
    )
