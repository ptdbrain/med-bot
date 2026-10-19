from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.ocr_benchmark import run_ocr_benchmark
from app.evaluation.ocr_metrics import (
    character_error_rate,
    diacritic_metrics,
    quote_containment,
    word_error_rate,
)
from scripts.run_ocr_benchmark import main as run_ocr_cli


def test_ocr_metrics_exact_match():
    ref = "Enzyme làm giảm năng lượng hoạt hóa."
    hyp = "Enzyme làm giảm năng lượng hoạt hóa."
    assert character_error_rate(ref, hyp) == 0.0
    assert word_error_rate(ref, hyp) == 0.0
    diacritics = diacritic_metrics(ref, hyp)
    assert diacritics["diacritic_preservation_rate"] == 1.0
    assert diacritics["diacritic_error_rate"] == 0.0

    containment = quote_containment(ref, hyp)
    assert containment["exact_contained"] is True
    assert containment["passed"] is True


def test_ocr_metrics_detect_diacritic_loss():
    ref = "làm giảm năng lượng hoạt hóa"
    hyp = "lam giam nang luong hoat hoa"  # Stripped diacritics
    diacritics = diacritic_metrics(ref, hyp)
    assert diacritics["diacritic_char_count"] > 0
    assert diacritics["diacritic_preservation_rate"] == 0.0
    assert diacritics["diacritic_error_rate"] == 1.0

    containment = quote_containment(ref, hyp)
    assert containment["exact_contained"] is False
    assert containment["diacritic_insensitive_contained"] is True


def test_ocr_metrics_word_and_char_errors():
    ref = "ATP transfers cellular energy"
    hyp = "ATP transfer cell energy"
    cer = character_error_rate(ref, hyp)
    wer = word_error_rate(ref, hyp)
    assert 0.0 < cer < 0.5
    assert 0.0 < wer <= 0.5


def test_run_ocr_benchmark_evaluates_corpus(tmp_path: Path):
    output_path = tmp_path / "ocr-report.json"
    report = run_ocr_benchmark(
        manifest_path=Path("evaluation/corpus/manifest.json"),
        transcripts_dir=Path("evaluation/corpus/ocr_transcripts"),
        output=output_path,
        cer_threshold=0.15,
    )
    assert report["benchmark"] == "ocr_extraction_benchmark"
    assert report["summary"]["total_documents"] == 12
    assert report["summary"]["passed_documents"] == 12
    assert report["summary"]["mean_cer"] <= 0.15
    assert report["summary"]["quote_containment_rate"] == 1.0
    assert output_path.is_file()

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["summary"]["total_documents"] == 12


def test_run_ocr_benchmark_cli(monkeypatch, tmp_path: Path):
    output_path = tmp_path / "cli-ocr-report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_ocr_benchmark.py",
            "--manifest",
            "evaluation/corpus/manifest.json",
            "--transcripts-dir",
            "evaluation/corpus/ocr_transcripts",
            "--output",
            str(output_path),
        ],
    )
    exit_code = run_ocr_cli()
    assert exit_code == 0
    assert output_path.is_file()
