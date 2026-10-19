from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import requests

from app.config import Settings


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    status: str
    message: str


def check_runtime(settings: Settings, *, probe_models: bool = False) -> list[RuntimeCheck]:
    checks = [RuntimeCheck("profile", "ok", settings.profile)]
    dependencies = {}
    if settings.embedding_provider == "sentence-transformers" or settings.reranker_provider == "cross-encoder":
        dependencies["sentence_transformers"] = "embedding/reranker"
    if settings.ocr_provider == "paddle":
        dependencies.update(paddleocr="ocr", pypdfium2="renderer")
    for module, stage in dependencies.items():
        exists = importlib.util.find_spec(module) is not None
        checks.append(RuntimeCheck(stage, "ok" if exists else "missing", module))
    if settings.embedding_provider == "sentence-transformers":
        model = Path(settings.embedding_model)
        if model.is_absolute() or settings.embedding_model.startswith(("data/", "./", "data\\")):
            present = model.is_dir() and (model / "config.json").is_file() and any(model.glob("*.safetensors"))
            checks.append(RuntimeCheck("embedding_model", "ok" if present else "missing", "Local model files"))
        else:
            checks.append(RuntimeCheck("embedding_model", "not_checked", "Remote model cache not verified"))
    if "http" in {
        settings.llm_provider,
        settings.embedding_provider,
        settings.ocr_provider,
        settings.reranker_provider,
    }:
        configured = bool(settings.http_provider_url and settings.http_provider_key)
        checks.append(
            RuntimeCheck("http_provider", "ok" if configured else "missing", "Endpoint and credentials configured")
        )
    if settings.llm_provider == "ollama":
        if not probe_models:
            checks.append(RuntimeCheck("generator", "not_checked", "Use --probe-models to check Ollama"))
        else:
            try:
                response = requests.get(settings.ollama_base_url + "/api/tags", timeout=2)
                response.raise_for_status()
                names = {item.get("name") for item in response.json().get("models", [])}
                present = settings.ollama_model in names
                checks.append(RuntimeCheck("generator", "ok" if present else "missing", "Configured Ollama model"))
            except (requests.RequestException, ValueError, TypeError):
                checks.append(RuntimeCheck("generator", "missing", "Ollama is unavailable"))
    if settings.vector_store_provider == "qdrant":
        if not probe_models:
            checks.append(RuntimeCheck("vector_store", "not_checked", "Qdrant not probed"))
        else:
            try:
                from app.retrieval.index_manifest import embedding_fingerprint

                collection = settings.qdrant_collection + "_" + embedding_fingerprint(settings)[:16]
                response = requests.get(settings.qdrant_url + "/collections/" + collection, timeout=2)
                response.raise_for_status()
                checks.append(RuntimeCheck("vector_store", "ok", "Qdrant collection reachable"))
            except requests.RequestException:
                checks.append(RuntimeCheck("vector_store", "missing", "Qdrant collection unavailable"))
    return checks
