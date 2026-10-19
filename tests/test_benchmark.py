from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.evaluation.benchmark import run_benchmark
from scripts.run_benchmark import main


def test_benchmark_rejects_unreviewed_cases_without_allow_draft(tmp_path: Path):
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["bm25"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid evaluation dataset.*unreviewed"):
        run_benchmark(config_path, "dev", output_path, allow_draft=False, limit=2)


def test_benchmark_runs_with_allow_draft_and_retains_not_evaluated_status(tmp_path: Path):
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["bm25"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    report = run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=2)

    assert report["quality_gate"] == "not_evaluated_unreviewed"
    assert report["dataset_reviewed"] is False
    assert report["faithfulness"]["status"] == "not_evaluated"
    assert output_path.is_file()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["quality_gate"] == "not_evaluated_unreviewed"
    assert saved["dataset_reviewed"] is False


def test_benchmark_rejects_dense_mode_with_lexical_provider(tmp_path: Path):
    config_path = tmp_path / "misleading_dense.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["dense"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Configuration error.*Mode 'dense' requested.*lexical"):
        run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=1)


def test_benchmark_rejects_hybrid_mode_with_lexical_provider(tmp_path: Path):
    config_path = tmp_path / "misleading_hybrid.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["hybrid"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Configuration error.*Mode 'hybrid' requested.*lexical"):
        run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=1)


def test_benchmark_rejects_rerank_mode_with_lexical_provider(tmp_path: Path):
    config_path = tmp_path / "misleading_rerank.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["rerank"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Configuration error.*Mode 'rerank' requested.*lexical"):
        run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=1)


def test_benchmark_accepts_lexical_mode_names_in_demo_regression(tmp_path: Path):
    config_path = tmp_path / "demo_regression.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["bm25", "lexical_vector", "lexical_hybrid", "lexical_rerank"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    report = run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=2)
    assert report["benchmark_type"] == "demo_regression"
    assert "bm25" in report["summary"]
    assert "lexical_vector" in report["summary"]
    assert "lexical_hybrid" in report["summary"]
    assert "lexical_rerank" in report["summary"]
    assert report["summary"]["lexical_vector"]["cases"] == 2


def test_benchmark_rejects_qdrant_when_unreachable(tmp_path: Path):
    config_path = tmp_path / "local_quality.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "local",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "qdrant",
                    "QDRANT_URL": "http://127.0.0.1:59999",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["dense"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires a running Qdrant instance"):
        run_benchmark(config_path, "dev", output_path, allow_draft=True, limit=1)


def test_run_benchmark_cli_distinguishes_draft_and_release(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "output.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/corpus/manifest.json",
                "cases": "evaluation/cases",
                "modes": ["bm25"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    # Without --allow-draft on unreviewed data, ValueError is raised
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_benchmark.py",
            "--config",
            str(config_path),
            "--split",
            "dev",
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="unreviewed"):
        main()

    # With --allow-draft, succeeds with exit code 0
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_benchmark.py",
            "--config",
            str(config_path),
            "--split",
            "dev",
            "--output",
            str(output_path),
            "--allow-draft",
            "--limit",
            "1",
        ],
    )
    exit_code = main()
    assert exit_code == 0
