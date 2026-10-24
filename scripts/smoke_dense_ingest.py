from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sqlite3
import struct
import sys
import time
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.sentence_transformer import SentenceTransformerEmbedding
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.dense_store import SQLiteDenseVectorStore
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.runtime.doctor import _cached_model_path
from scripts.check_index import inspect_index

FIXTURES = {
    "storage.md": (
        "# Quy ước lưu trữ tài liệu thử nghiệm\n\n"
        "Mã kiểm thử lưu trữ là SMOKE-2026. Tài liệu được giữ trong 30 ngày. "
        "Đây là dữ liệu tổng hợp để kiểm tra ingestion, không chứa hồ sơ người bệnh.\n"
    ),
    "weather.md": (
        "# Quan sát khí tượng\n\n"
        "Trạm khí tượng ghi lại nhiệt độ và lượng mưa mỗi buổi sáng. "
        "Bản ghi thời tiết chỉ phục vụ kiểm tra tìm kiếm, không mô tả quy ước lưu trữ tài liệu.\n"
    ),
}
QUERIES = (
    ("vi", "Mã kiểm thử lưu trữ tài liệu là gì?"),
    ("en", "What is the storage test code and how long are the documents kept?"),
)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("torch", "sentence-transformers", "transformers"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not_installed"
    return versions


def _validate_vectors(path: Path, dimension: int) -> dict:
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as connection:
        rows = connection.execute("SELECT vector FROM dense_vectors").fetchall()
    invalid = 0
    norms: list[float] = []
    for (blob,) in rows:
        if len(blob) != dimension * 4:
            invalid += 1
            continue
        values = struct.unpack(f"<{dimension}f", blob)
        if not all(math.isfinite(value) for value in values):
            invalid += 1
            continue
        norm = math.sqrt(sum(value * value for value in values))
        norms.append(norm)
        invalid += int(norm == 0)
    return {"passed": bool(rows) and invalid == 0, "count": len(rows), "invalid": invalid, "norms": norms}


def run_smoke(
    run_dir: Path,
    model_name: str = "BAAI/bge-m3",
    *,
    embedding_provider: EmbeddingProvider | None = None,
    dimension: int = 1024,
) -> dict:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    settings = Settings(
        data_dir=run_dir,
        index_path=run_dir / "indexes" / "unused-lexical.json",
        metadata_db_path=run_dir / "indexes" / "metadata.db",
        dense_index_path=run_dir / "indexes" / "dense.db",
        embedding_provider="sentence-transformers",
        embedding_model=model_name,
        embedding_dimension=dimension,
        vector_store_provider="sqlite-dense",
    )
    report = {
        "scope": "synthetic_markdown_dense_ingestion_smoke_not_release_quality",
        "passed": False,
        "run_dir": str(run_dir),
        "started_at": datetime.now(UTC).isoformat(),
        "embedding_execution": "injected_test_provider" if embedding_provider is not None else "local_model_inference",
        "embedding_model": model_name,
        "dimension": dimension,
        "vector_store": "sqlite-dense",
        "generation": "not_run",
        "reranking": "not_run",
        "ocr": "not_run",
        "qdrant": "not_run",
        "active_revision": "not_implemented",
        "versions": _versions(),
        "platform": platform.platform(),
        "sources_sha256": {name: hashlib.sha256(text.encode("utf-8")).hexdigest() for name, text in FIXTURES.items()},
        "queries": [],
        "errors": [],
    }
    started = time.perf_counter()
    stage = "load_embedding"
    try:
        if embedding_provider is None:
            model_path = _cached_model_path(model_name)
            if model_path is None:
                raise RuntimeError("Model must already be available locally; no model downloads are performed.")
            report["model_snapshot"] = str(model_path)
            embedding_provider = SentenceTransformerEmbedding(
                str(model_path), device="cpu", batch_size=2, local_files_only=True
            )

        metadata = SQLiteMetadataStore(settings.metadata_db_path)
        vectors = SQLiteDenseVectorStore(settings.dense_index_path, embedding_provider, dimension)
        pipeline = IngestionPipeline(settings, vectors, metadata)
        stage = "ingest"
        ingest_started = time.perf_counter()
        ingested = [pipeline.ingest_file(name, text.encode("utf-8")) for name, text in FIXTURES.items()]
        report["ingest_seconds"] = time.perf_counter() - ingest_started
        report["ingested"] = [asdict(result) for result in ingested]

        stage = "validate_persisted_index"
        report["index_check"] = inspect_index(
            settings.metadata_db_path, settings.dense_index_path, settings.index_path, "sqlite-dense"
        )
        report["vector_validation"] = _validate_vectors(settings.dense_index_path, dimension)
        if not report["index_check"]["passed"] or not report["vector_validation"]["passed"]:
            raise RuntimeError("Persisted vector validation failed.")

        # New store instances check persistence without re-ingesting or re-embedding documents.
        stage = "reopen_and_retrieve"
        reopened_metadata = SQLiteMetadataStore(settings.metadata_db_path)
        reopened_vectors = SQLiteDenseVectorStore(settings.dense_index_path, embedding_provider, dimension)
        report["documents"] = len(reopened_metadata.list_documents())
        expected_doc_id = ingested[0].doc_id
        for language, query in QUERIES:
            query_started = time.perf_counter()
            results = reopened_vectors.search(query, top_k=2)
            best = results[0] if results else None
            passed = bool(best and best.chunk.metadata.doc_id == expected_doc_id and "SMOKE-2026" in best.chunk.text)
            report["queries"].append(
                {
                    "language": language,
                    "query": query,
                    "passed": passed,
                    "after_store_reopen": True,
                    "evidence": best.chunk.text if best else "",
                    "chunk_id": best.chunk.chunk_id if best else None,
                    "score": best.score if best else None,
                    "seconds": time.perf_counter() - query_started,
                }
            )
        report["passed"] = report["documents"] == 2 and all(case["passed"] for case in report["queries"])
        if not report["passed"]:
            report["errors"].append("expected_evidence_not_retrieved")
    except Exception as exc:  # noqa: BLE001 - persist a failed live-run artifact, not a success or silent fallback.
        report["errors"].append({"stage": stage, "type": type(exc).__name__})
    report["total_seconds"] = time.perf_counter() - started
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run isolated dense Markdown ingestion smoke using a cached CPU model")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    run_dir = args.run_dir or Path("data/processed/runtime-smoke") / uuid4().hex
    try:
        report = run_smoke(run_dir, args.model)
    except FileExistsError:
        parser.error("The run directory already exists; choose a new one to preserve previous data.")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(f"Dense ingest smoke: {'PASS' if report['passed'] else 'FAIL'}")
        print(f"Report: {run_dir.resolve() / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
