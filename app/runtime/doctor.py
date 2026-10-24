from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import requests

from app.config import Settings

CheckStatus = Literal["ok", "missing", "error", "not_checked"]


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    status: CheckStatus
    message: str


def is_ready(checks: list[RuntimeCheck]) -> bool:
    return all(check.status not in {"missing", "error"} for check in checks)


def _package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _cached_model_path(model_name: str) -> Path | None:
    model_path = Path(model_name).expanduser()
    if model_path.is_dir():
        return model_path if _model_files_present(model_path) else None

    cache_root = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
    hub = Path(os.getenv("HF_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", cache_root / "hub")))
    model_cache = hub / f"models--{model_name.replace('/', '--')}"
    snapshots = model_cache / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir() and _model_files_present(path)]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def _model_files_present(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        weight = path / name
        if weight.is_file() and weight.stat().st_size > 0:
            return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        manifest = path / name
        if manifest.is_file():
            try:
                shards = set(json.loads(manifest.read_text(encoding="utf-8"))["weight_map"].values())
                return bool(shards) and all(
                    (path / shard).is_file() and (path / shard).stat().st_size > 0 for shard in shards
                )
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                return False
    return False


def _check_ollama(settings: Settings, probe_models: bool) -> list[RuntimeCheck]:
    try:
        response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=2)
    except requests.RequestException:
        return [
            RuntimeCheck("ollama", "missing", "Ollama is unreachable at the configured local endpoint."),
            RuntimeCheck("ollama_model", "not_checked", "Ollama service is unavailable."),
        ]
    if response.status_code != 200:
        return [
            RuntimeCheck("ollama", "error", f"Ollama returned HTTP {response.status_code}."),
            RuntimeCheck("ollama_model", "not_checked", "Model listing failed."),
        ]

    models = response.json().get("models", [])
    model_names = {str(item.get("name", item.get("model", ""))) for item in models}
    if settings.ollama_model not in model_names:
        return [
            RuntimeCheck("ollama", "ok", "Ollama service responded."),
            RuntimeCheck("ollama_model", "missing", f"Configured model '{settings.ollama_model}' is not installed."),
        ]

    checks = [
        RuntimeCheck("ollama", "ok", "Ollama service responded."),
        RuntimeCheck("ollama_model", "ok", f"Configured model '{settings.ollama_model}' is installed."),
    ]
    if probe_models:
        try:
            result = requests.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": "Reply with the single word OK.",
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 8},
                },
                timeout=settings.ollama_timeout_seconds,
            )
            result.raise_for_status()
            generated = result.json().get("response")
            valid_output = isinstance(generated, str) and bool(generated.strip())
            status: CheckStatus = "ok" if valid_output else "error"
            message = (
                "Real generation probe returned output." if valid_output else "Generation probe returned no valid text."
            )
        except requests.RequestException:
            status, message = "error", "Ollama generation probe failed or timed out."
        checks.append(RuntimeCheck("ollama_inference", status, message))
    else:
        checks.append(RuntimeCheck("ollama_inference", "not_checked", "Use --probe-models to run generation."))
    return checks


def _check_qdrant(settings: Settings) -> list[RuntimeCheck]:
    base_url = settings.qdrant_url.rstrip("/")
    try:
        health = requests.get(f"{base_url}/healthz", timeout=2)
    except requests.RequestException:
        return [
            RuntimeCheck("qdrant", "missing", "Qdrant is unreachable at the configured endpoint."),
            RuntimeCheck("qdrant_collection", "not_checked", "Qdrant service is unavailable."),
        ]
    if health.status_code != 200:
        return [
            RuntimeCheck("qdrant", "error", f"Qdrant returned HTTP {health.status_code} on /healthz."),
            RuntimeCheck("qdrant_collection", "not_checked", "Qdrant health check failed."),
        ]

    encoded_collection = quote(settings.qdrant_collection, safe="")
    try:
        collection = requests.get(f"{base_url}/collections/{encoded_collection}", timeout=2)
    except requests.RequestException:
        return [
            RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
            RuntimeCheck("qdrant_collection", "error", "Collection probe failed or timed out."),
        ]
    if collection.status_code == 404:
        return [
            RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
            RuntimeCheck("qdrant_collection", "missing", f"Collection '{settings.qdrant_collection}' does not exist."),
        ]
    if collection.status_code != 200:
        return [
            RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
            RuntimeCheck("qdrant_collection", "error", f"Collection probe returned HTTP {collection.status_code}."),
        ]

    vectors = collection.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
    if not isinstance(vectors, dict) or "size" not in vectors:
        return [
            RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
            RuntimeCheck("qdrant_collection", "error", "The configured adapter requires an unnamed dense vector."),
        ]
    if int(vectors["size"]) != settings.embedding_dimension:
        return [
            RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
            RuntimeCheck("qdrant_collection", "error", "Collection vector size does not match EMBEDDING_DIMENSION."),
        ]
    return [
        RuntimeCheck("qdrant", "ok", "Qdrant service responded."),
        RuntimeCheck("qdrant_collection", "ok", f"Collection '{settings.qdrant_collection}' is available."),
    ]


def _check_embedding(settings: Settings, probe_models: bool) -> RuntimeCheck:
    provider = settings.embedding_provider.strip().lower()
    if provider in {"lexical", "local"}:
        return RuntimeCheck("embedding_model", "not_checked", "Lexical mode does not use dense embeddings.")
    if provider != "sentence-transformers":
        return RuntimeCheck("embedding_model", "error", f"Unsupported embedding provider '{provider}'.")
    if not _package_available("sentence_transformers"):
        return RuntimeCheck("embedding_model", "missing", "Install sentence-transformers to use dense embeddings.")
    model_path = _cached_model_path(settings.embedding_model)
    if model_path is None:
        return RuntimeCheck(
            "embedding_model", "missing", "Embedding model is not fully cached; doctor will not download it."
        )
    if not probe_models:
        return RuntimeCheck(
            "embedding_model", "ok", "Embedding dependency and local model cache are present; inference not run."
        )

    try:
        from app.embeddings.sentence_transformer import SentenceTransformerEmbedding

        vector = SentenceTransformerEmbedding(
            str(model_path), settings.embedding_device, local_files_only=True
        ).embed_query("runtime doctor probe")
        if len(vector) != settings.embedding_dimension or not all(math.isfinite(value) for value in vector):
            return RuntimeCheck("embedding_model", "error", "Embedding inference returned invalid values or dimension.")
    except Exception as exc:  # noqa: BLE001 - model initialization errors must be reported as a probe result.
        return RuntimeCheck("embedding_model", "error", f"Embedding inference failed ({type(exc).__name__}).")
    return RuntimeCheck("embedding_model", "ok", f"Real embedding inference passed at dimension {len(vector)}.")


def _check_reranker(settings: Settings, probe_models: bool) -> RuntimeCheck:
    provider = settings.reranker_provider.strip().lower()
    if provider in {"lexical", "local"}:
        return RuntimeCheck("reranker_model", "not_checked", "Lexical reranking does not load a model.")
    if provider not in {"cross-encoder", "cross_encoder"}:
        return RuntimeCheck("reranker_model", "error", f"Unsupported reranker provider '{provider}'.")
    if not _package_available("sentence_transformers"):
        return RuntimeCheck("reranker_model", "missing", "Install sentence-transformers to use the cross-encoder.")
    model_path = _cached_model_path(settings.reranker_model)
    if model_path is None:
        return RuntimeCheck(
            "reranker_model", "missing", "Reranker model is not fully cached; doctor will not download it."
        )
    if not probe_models:
        return RuntimeCheck(
            "reranker_model", "ok", "Reranker dependency and local model cache are present; inference not run."
        )

    try:
        from sentence_transformers import CrossEncoder

        scores = CrossEncoder(str(model_path), device=settings.embedding_device, local_files_only=True).predict(
            [["runtime doctor query", "runtime doctor evidence"]],
            show_progress_bar=False,
        )
        if len(scores) != 1 or not math.isfinite(float(scores[0])):
            return RuntimeCheck("reranker_model", "error", "Cross-encoder inference returned an invalid score.")
    except Exception as exc:  # noqa: BLE001 - model initialization errors must be reported as a probe result.
        return RuntimeCheck("reranker_model", "error", f"Cross-encoder inference failed ({type(exc).__name__}).")
    return RuntimeCheck("reranker_model", "ok", "Real cross-encoder inference passed.")


def check_runtime(settings: Settings, *, probe_models: bool = False) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    if settings.llm_provider.strip().lower() == "ollama":
        try:
            checks.extend(_check_ollama(settings, probe_models))
        except (ValueError, KeyError, TypeError, AttributeError):
            checks.append(RuntimeCheck("ollama", "error", "Ollama returned an invalid response payload."))
    elif settings.llm_provider.strip().lower() == "mock":
        checks.append(RuntimeCheck("ollama", "not_checked", "Mock LLM configured; this is not real generation."))
    else:
        checks.append(RuntimeCheck("ollama", "error", f"Unsupported local LLM provider '{settings.llm_provider}'."))

    checks.append(_check_embedding(settings, probe_models))
    vector_provider = settings.vector_store_provider.strip().lower()
    dense_provider = settings.embedding_provider.strip().lower() == "sentence-transformers"
    dense_store = vector_provider in {"qdrant", "sqlite-dense", "dense-local"}
    if dense_provider != dense_store:
        checks.append(
            RuntimeCheck(
                "provider_compatibility", "error", "Dense embeddings and dense vector store must be selected together."
            )
        )
    if vector_provider == "qdrant":
        try:
            checks.extend(_check_qdrant(settings))
        except (ValueError, KeyError, TypeError, AttributeError):
            checks.append(RuntimeCheck("qdrant_collection", "error", "Qdrant returned an invalid response payload."))
    elif vector_provider in {"sqlite-dense", "dense-local"}:
        checks.append(RuntimeCheck("vector_store", "ok", "SQLite dense store is configured."))
    elif vector_provider in {"memory", "local"}:
        checks.append(RuntimeCheck("vector_store", "not_checked", "In-memory/local lexical store is configured."))
    else:
        checks.append(RuntimeCheck("vector_store", "error", f"Unsupported vector-store provider '{vector_provider}'."))

    checks.append(_check_reranker(settings, probe_models))
    if settings.profile_name == "local":
        checks.append(
            RuntimeCheck("pdf_ocr", "missing", "Scanned-PDF OCR is not implemented by the current PDF loader.")
        )
    return checks
