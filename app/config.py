from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import get_type_hints


@dataclass(frozen=True)
class Settings:
    profile: str = "demo"
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
    ocr_provider: str = "none"
    ocr_languages: tuple[str, ...] = ("vi", "en")
    ocr_dpi: int = 160
    ocr_max_pixels: int = 2_000_000
    pdf_max_pages: int = 1000
    pdf_max_pixels: int = 25_000_000
    chunk_unit: str = "characters"
    chunk_target_tokens: int = 500
    chunk_overlap_tokens: int = 80
    chunk_min_tokens: int = 100
    chunk_max_tokens: int = 640
    embedding_revision: str = "main"
    dense_candidate_k: int = 40
    sparse_candidate_k: int = 40
    relevance_dense_threshold: float = 0.45
    relevance_reranker_threshold: float = 0.5
    context_window_tokens: int = 8192
    context_output_reserve: int = 1024
    answer_validation: str = "basic"
    http_provider_url: str = ""
    http_provider_key: str = ""
    http_provider_timeout: int = 60
    http_embedding_fingerprint: str = ""


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


def _legacy_settings() -> Settings:
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


PROFILE_DEFAULTS = {
    "demo": {},
    "local": {
        "LLM_PROVIDER": "ollama",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "VECTOR_STORE_PROVIDER": "qdrant",
        "RERANKER_PROVIDER": "cross-encoder",
        "OCR_PROVIDER": "paddle",
        "CHUNK_UNIT": "tokens",
        "ANSWER_VALIDATION": "strict",
    },
    "cloud": {
        "LLM_PROVIDER": "http",
        "EMBEDDING_PROVIDER": "http",
        "OCR_PROVIDER": "http",
        "RERANKER_PROVIDER": "http",
        "VECTOR_STORE_PROVIDER": "qdrant",
        "CHUNK_UNIT": "tokens",
        "ANSWER_VALIDATION": "strict",
    },
    "hybrid": {
        "LLM_PROVIDER": "http",
        "EMBEDDING_PROVIDER": "sentence-transformers",
        "OCR_PROVIDER": "paddle",
        "RERANKER_PROVIDER": "cross-encoder",
        "VECTOR_STORE_PROVIDER": "qdrant",
        "CHUNK_UNIT": "tokens",
        "ANSWER_VALIDATION": "strict",
    },
}


def resolve_profile(name: str, overrides: dict[str, str]) -> Settings:
    if name not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown RAG profile: {name}")
    env = {**PROFILE_DEFAULTS[name], **{k: v for k, v in overrides.items() if v != ""}}
    types = get_type_hints(Settings)
    values = {"profile": name}
    for item in fields(Settings):
        value = env.get(item.name.upper())
        if value is None or item.name == "profile":
            continue
        typ = types[item.name]
        if typ is bool:
            if value.lower() not in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
                raise ValueError(f"Invalid boolean: {item.name}")
            values[item.name] = value.lower() in {"true", "1", "yes", "on"}
        elif typ in (int, float, Path):
            values[item.name] = typ(value)
        elif item.name in {"frontend_origins", "ocr_languages"}:
            values[item.name] = tuple(v.strip() for v in value.split(",") if v.strip())
        else:
            values[item.name] = value
    data = Path(values.get("data_dir", "data"))
    for key, filename in (
        ("index_path", "vector_store.json"),
        ("dense_index_path", "dense_vectors.db"),
        ("metadata_db_path", "metadata.db"),
    ):
        values.setdefault(key, data / "indexes" / filename)
    values["ollama_base_url"] = str(values.get("ollama_base_url", Settings.ollama_base_url)).rstrip("/")
    settings = Settings(**values)
    for key in (
        "embedding_batch_size",
        "embedding_dimension",
        "chunk_size",
        "pdf_max_pages",
        "pdf_max_pixels",
        "ocr_dpi",
        "chunk_target_tokens",
        "chunk_max_tokens",
        "dense_candidate_k",
        "sparse_candidate_k",
        "ollama_num_predict",
        "ocr_max_pixels",
    ):
        if getattr(settings, key) <= 0:
            raise ValueError(f"{key} must be positive")
    if not 0 <= settings.chunk_overlap_tokens < settings.chunk_target_tokens <= settings.chunk_max_tokens:
        raise ValueError("Invalid token chunk window")
    return settings


def get_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
    return resolve_profile(os.getenv("RAG_PROFILE", "demo"), dict(os.environ))
