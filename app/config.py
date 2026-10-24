from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    profile_name: str = "demo"
    project_name: str = "MedTutor Multimodal RAG Copilot"
    llm_provider: str = "mock"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    ollama_timeout_seconds: int = 180
    ollama_think: bool = False
    ollama_num_predict: int = 512
    embedding_provider: str = "lexical"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8
    chunk_size: int = 900
    chunk_overlap: int = 120
    chunk_min_size: int = 180
    hybrid_candidate_multiplier: int = 4
    hybrid_rrf_k: int = 60
    hybrid_dense_weight: float = 1.0
    hybrid_sparse_weight: float = 1.0
    reranker_provider: str = "lexical"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_candidate_k: int = 20
    reranker_batch_size: int = 8
    reranker_diversity_lambda: float = 0.75
    vector_store_provider: str = "memory"
    dense_write_batch_size: int = 64
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
    dense_index_path: Path = Path("data/indexes/dense_vectors.db")
    metadata_db_path: Path = Path("data/indexes/metadata.db")


_PROFILE_DEFAULTS: dict[str, dict[str, object]] = {
    "demo": {
        "llm_provider": "mock",
        "use_mock_llm": True,
        "embedding_provider": "lexical",
        "vector_store_provider": "memory",
        "reranker_provider": "lexical",
    },
    "local": {
        "llm_provider": "ollama",
        "use_mock_llm": False,
        "embedding_provider": "sentence-transformers",
        "vector_store_provider": "qdrant",
        "reranker_provider": "cross-encoder",
    },
}

_PROFILE_ENV_FIELDS = {
    "LLM_PROVIDER": "llm_provider",
    "OLLAMA_BASE_URL": "ollama_base_url",
    "OLLAMA_MODEL": "ollama_model",
    "OLLAMA_TIMEOUT_SECONDS": "ollama_timeout_seconds",
    "EMBEDDING_PROVIDER": "embedding_provider",
    "EMBEDDING_MODEL": "embedding_model",
    "EMBEDDING_DEVICE": "embedding_device",
    "EMBEDDING_DIMENSION": "embedding_dimension",
    "VECTOR_STORE_PROVIDER": "vector_store_provider",
    "QDRANT_URL": "qdrant_url",
    "QDRANT_COLLECTION": "qdrant_collection",
    "RERANKER_PROVIDER": "reranker_provider",
    "RERANKER_MODEL": "reranker_model",
    "USE_MOCK_LLM": "use_mock_llm",
    "DATA_DIR": "data_dir",
    "DENSE_INDEX_PATH": "dense_index_path",
}


def _parse_profile_value(field_name: str, value: str) -> object:
    if field_name in {"data_dir", "dense_index_path"}:
        return Path(value)
    if field_name == "use_mock_llm":
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if field_name == "embedding_dimension":
        return int(value)
    if field_name == "ollama_timeout_seconds":
        return int(value)
    return value.rstrip("/") if field_name.endswith("_url") else value


def resolve_profile(name: str, overrides: dict[str, str] | None = None) -> Settings:
    profile = name.strip().lower()
    if profile not in _PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile '{name}'. Supported profiles: {', '.join(sorted(_PROFILE_DEFAULTS))}.")
    unknown = set(overrides or {}) - _PROFILE_ENV_FIELDS.keys()
    if unknown:
        raise ValueError(f"Unknown profile override keys: {', '.join(sorted(unknown))}.")

    values = dict(_PROFILE_DEFAULTS[profile])
    for source in (os.environ, overrides or {}):
        for env_name, field_name in _PROFILE_ENV_FIELDS.items():
            value = source.get(env_name)
            if value is not None and value.strip():
                values[field_name] = _parse_profile_value(field_name, value)

    if overrides and overrides.get("DATA_DIR"):
        root = Path(overrides["DATA_DIR"])
        values.update(
            index_path=root / "indexes" / "vector_store.json",
            metadata_db_path=root / "indexes" / "metadata.db",
        )
        if not overrides.get("DENSE_INDEX_PATH"):
            values["dense_index_path"] = root / "indexes" / "dense_vectors.db"
    return replace(_settings_from_environment(), profile_name=profile, **values)


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
    return resolve_profile(os.getenv("RAG_PROFILE", "demo"))


def _settings_from_environment() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/"),
        ollama_model=os.getenv(
            "OLLAMA_MODEL",
            "qwen3:4b-instruct-2507-q4_K_M",
        ),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        ollama_think=_bool_env("OLLAMA_THINK", False),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "lexical"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        embedding_device=os.getenv("EMBEDDING_DEVICE", "cpu"),
        embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "120")),
        chunk_min_size=int(os.getenv("CHUNK_MIN_SIZE", "180")),
        hybrid_candidate_multiplier=int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4")),
        hybrid_rrf_k=int(os.getenv("HYBRID_RRF_K", "60")),
        hybrid_dense_weight=float(os.getenv("HYBRID_DENSE_WEIGHT", "1.0")),
        hybrid_sparse_weight=float(os.getenv("HYBRID_SPARSE_WEIGHT", "1.0")),
        reranker_provider=os.getenv("RERANKER_PROVIDER", "lexical"),
        reranker_model=os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        ),
        reranker_candidate_k=int(os.getenv("RERANKER_CANDIDATE_K", "20")),
        reranker_batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "8")),
        reranker_diversity_lambda=float(os.getenv("RERANKER_DIVERSITY_LAMBDA", "0.75")),
        vector_store_provider=os.getenv("VECTOR_STORE_PROVIDER", "memory"),
        dense_write_batch_size=int(os.getenv("DENSE_WRITE_BATCH_SIZE", "64")),
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
        dense_index_path=Path(
            os.getenv(
                "DENSE_INDEX_PATH",
                str(data_dir / "indexes" / "dense_vectors.db"),
            )
        ),
        metadata_db_path=data_dir / "indexes" / "metadata.db",
    )
