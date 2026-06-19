from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    project_name: str = "MedTutor Multimodal RAG Copilot"
    llm_provider: str = "mock"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: int = 180
    embedding_provider: str = "lexical"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8
    vector_store_provider: str = "memory"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "medtutor_chunks"
    database_url: str = "sqlite:///data/indexes/metadata.db"
    use_mock_llm: bool = True
    use_local_embeddings: bool = True
    stt_provider: str = "mock"
    mock_stt_transcript: str = "Đây là transcript mẫu cho chế độ demo."
    stt_language: str | None = "vi"
    whisper_model_size: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    max_upload_bytes: int = 15 * 1024 * 1024
    frontend_origins: tuple[str, ...] = ("http://localhost:3000",)
    data_dir: Path = Path("data")
    index_path: Path = Path("data/indexes/vector_store.json")
    metadata_db_path: Path = Path("data/indexes/metadata.db")


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.getenv(name)
    if value is None:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "lexical"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "memory"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "medtutor_chunks"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///data/indexes/metadata.db"),
        use_mock_llm=_bool_env("USE_MOCK_LLM", True),
        use_local_embeddings=_bool_env("USE_LOCAL_EMBEDDINGS", True),
        stt_provider=os.getenv("STT_PROVIDER", "mock"),
        mock_stt_transcript=os.getenv(
            "MOCK_STT_TRANSCRIPT",
            "Đây là transcript mẫu cho chế độ demo.",
        ),
        stt_language=os.getenv("STT_LANGUAGE", "vi") or None,
        whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "small"),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(15 * 1024 * 1024))),
        frontend_origins=_csv_env(
            "FRONTEND_ORIGINS",
            ("http://localhost:3000",),
        ),
        data_dir=data_dir,
        index_path=data_dir / "indexes" / "vector_store.json",
        metadata_db_path=data_dir / "indexes" / "metadata.db",
    )
