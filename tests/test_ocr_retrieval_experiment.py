from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.benchmark import run_benchmark


def test_ocr_retrieval_slice_runs_and_records_corpus_source(tmp_path: Path):
    clean_output = tmp_path / "clean-report.json"
    ocr_output = tmp_path / "ocr-report.json"

    # 1. Clean run
    clean_config = tmp_path / "clean_config.json"
    clean_config.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data_clean"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "corpus_source": "clean",
                "cases": "evaluation/cases",
                "modes": ["bm25", "lexical_vector"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )
    clean_report = run_benchmark(clean_config, "dev", clean_output, allow_draft=True, limit=4)
    assert clean_report["corpus_source"] == "clean"

    # 2. OCR-extracted run
    ocr_config = tmp_path / "ocr_config.json"
    ocr_config.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data_ocr"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "corpus_source": "ocr",
                "cases": "evaluation/cases",
                "modes": ["bm25", "lexical_vector"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )
    ocr_report = run_benchmark(ocr_config, "dev", ocr_output, allow_draft=True, limit=4)
    assert ocr_report["corpus_source"] == "ocr"

    # Both runs should complete cleanly without errors
    assert clean_report["summary"]["bm25"]["cases"] == 4
    assert ocr_report["summary"]["bm25"]["cases"] == 4
    assert clean_report["summary"]["bm25"]["failures"] == 0
    assert ocr_report["summary"]["bm25"]["failures"] == 0
