from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.ocr_benchmark import run_ocr_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run independent OCR benchmark on scanned PDF fixtures.")
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/corpus/manifest.json"))
    parser.add_argument("--scan-dir", type=Path, default=Path("evaluation/corpus"))
    parser.add_argument("--transcripts-dir", type=Path, default=Path("evaluation/corpus/ocr_transcripts"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/ocr-benchmark.json"))
    parser.add_argument("--cer-threshold", type=float, default=0.15)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--use-live-ocr", action="store_true", help="Attempt live OCR recognition with PaddleOCR")
    args = parser.parse_args()

    ocr_provider = None
    if args.use_live_ocr:
        try:
            from app.ingestion.ocr.paddle import PaddleOCRProvider

            ocr_provider = PaddleOCRProvider()
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Live OCR provider failed to initialize: {exc}. Falling back to transcripts.")

    report = run_ocr_benchmark(
        manifest_path=args.manifest,
        ocr_provider=ocr_provider,
        scan_dir=args.scan_dir,
        transcripts_dir=args.transcripts_dir,
        output=args.output,
        dpi=args.dpi,
        cer_threshold=args.cer_threshold,
    )

    print(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "total_documents": report["summary"]["total_documents"],
                "passed_documents": report["summary"]["passed_documents"],
                "mean_cer": round(report["summary"]["mean_cer"], 4),
                "mean_wer": round(report["summary"]["mean_wer"], 4),
                "diacritic_preservation_rate": round(report["summary"]["diacritic_preservation_rate"], 4),
                "quote_containment_rate": round(report["summary"]["quote_containment_rate"], 4),
                "pass_rate": round(report["summary"]["pass_rate"], 4),
            },
            indent=2,
        )
    )
    return int(report["summary"]["passed_documents"] < report["summary"]["total_documents"])


if __name__ == "__main__":
    raise SystemExit(main())
