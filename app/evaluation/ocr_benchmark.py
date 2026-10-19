from __future__ import annotations

import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

from app.evaluation.metrics import percentile
from app.evaluation.ocr_metrics import (
    character_error_rate,
    diacritic_metrics,
    quote_containment,
    word_error_rate,
)


def load_ocr_transcript(transcripts_dir: Path, key: str) -> dict[str, Any] | None:
    path = transcripts_dir / f"{key}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def run_ocr_benchmark(
    manifest_path: Path = Path("evaluation/corpus/manifest.json"),
    *,
    ocr_provider: Any = None,
    scan_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    output: Path | None = None,
    dpi: int = 160,
    cer_threshold: float = 0.15,
) -> dict[str, Any]:
    """Run an independent OCR benchmark decoupling extraction errors from retrieval.

    Measures CER, WER, Vietnamese diacritic preservation, quote containment, and latency
    for each scanned document against its ground truth reference.
    """
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    evidence_dict = manifest.get("evidence", manifest)

    if scan_dir is None:
        scan_dir = manifest_path.parent
    if transcripts_dir is None:
        transcripts_dir = manifest_path.parent / "ocr_transcripts"

    results: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for key, ev in sorted(evidence_dict.items()):
        ref_source_path = Path(ev["source_path"])
        if not ref_source_path.is_absolute():
            # Resolve relative to manifest if needed
            candidate = manifest_path.parent.parent.parent / ref_source_path
            if candidate.is_file():
                ref_source_path = candidate
            elif not ref_source_path.is_file():
                ref_source_path = manifest_path.parent / ref_source_path.name

        ref_text = ref_source_path.read_text(encoding="utf-8") if ref_source_path.is_file() else ev.get("quote", "")
        ref_quote = ev.get("quote", "")
        language = ev.get("language", "vi")

        scan_pdf_path = scan_dir / f"{key}-scan.pdf"
        if not scan_pdf_path.is_file():
            # Try alternate naming
            scan_pdf_path = scan_dir / f"{key}.pdf"

        hyp_text = ""
        extraction_method = "unknown"
        confidence: float | None = None
        started = time.perf_counter()
        status = "ok"
        error_msg = None

        try:
            if ocr_provider is not None and scan_pdf_path.is_file():
                from app.ingestion.pdf_loader import PdfLoader

                loader = PdfLoader(ocr=ocr_provider, dpi=dpi)
                doc = loader.load(scan_pdf_path, scan_pdf_path.read_bytes())
                hyp_text = doc.text
                extraction_method = "live_ocr"
                confidence = doc.metadata.get("extraction", {}).get("metrics", {}).get("ocr_confidence")
            elif transcripts_dir.is_dir() and (transcripts_dir / f"{key}.json").is_file():
                transcript_data = load_ocr_transcript(transcripts_dir, key)
                if transcript_data:
                    hyp_text = transcript_data.get("text", "")
                    extraction_method = transcript_data.get("method", "transcript_cache")
                    confidence = transcript_data.get("confidence")
            elif scan_pdf_path.is_file():
                # Attempt to extract with PdfLoader without OCR if native or fallback
                from app.ingestion.pdf_loader import PdfLoader

                loader = PdfLoader(ocr=None, dpi=dpi)
                try:
                    doc = loader.load(scan_pdf_path, scan_pdf_path.read_bytes())
                    hyp_text = doc.text
                    extraction_method = "native_pdf"
                except Exception as exc:  # noqa: BLE001
                    status = "failed"
                    error_msg = f"OCR provider required for scan PDF ({exc})"
            else:
                status = "failed"
                error_msg = f"No scan PDF or transcript found for {key}"
        except Exception as exc:  # noqa: BLE001
            status = "failed"
            error_msg = str(exc)

        elapsed = time.perf_counter() - started

        if status == "ok":
            # Strip internal loader page markers (e.g. <!-- page:1 -->) from evaluation text
            eval_hyp_text = re.sub(r"<!--\s*page:\d+\s*-->\n?", "", hyp_text).strip()
            # Also normalize heading markers from reference markdown
            eval_ref_text = re.sub(r"^#+\s*", "", ref_text.strip())

            cer = character_error_rate(eval_ref_text, eval_hyp_text)
            wer = word_error_rate(eval_ref_text, eval_hyp_text)
            diacritic = diacritic_metrics(eval_ref_text, eval_hyp_text)
            containment = quote_containment(ref_quote, hyp_text)
            passed = (cer <= cer_threshold) and containment["passed"]
        else:
            cer, wer = 1.0, 1.0
            diacritic = {
                "diacritic_char_count": 0,
                "diacritic_preservation_rate": 0.0,
                "diacritic_error_rate": 1.0,
            }
            containment = {
                "exact_contained": False,
                "normalized_contained": False,
                "diacritic_insensitive_contained": False,
                "token_recall": 0.0,
                "passed": False,
            }
            passed = False

        results.append(
            {
                "key": key,
                "language": language,
                "status": status,
                "error": error_msg,
                "extraction_method": extraction_method,
                "reference_char_count": len(ref_text),
                "hypothesis_char_count": len(hyp_text),
                "confidence": confidence,
                "cer": cer,
                "wer": wer,
                "diacritic_preservation_rate": diacritic["diacritic_preservation_rate"],
                "diacritic_error_rate": diacritic["diacritic_error_rate"],
                "quote_contained": containment["passed"],
                "quote_containment": containment,
                "seconds": elapsed,
                "passed": passed,
            }
        )

    # Compute summary aggregates
    total_elapsed = time.perf_counter() - total_start
    usable_results = [r for r in results if r["status"] == "ok"]
    vi_results = [r for r in usable_results if r["language"] == "vi"]
    en_results = [r for r in usable_results if r["language"] == "en"]

    summary = {
        "total_documents": len(results),
        "successful_extractions": len(usable_results),
        "passed_documents": sum(1 for r in results if r["passed"]),
        "pass_rate": (sum(1 for r in results if r["passed"]) / len(results)) if results else 0.0,
        "mean_cer": sum(r["cer"] for r in usable_results) / len(usable_results) if usable_results else 1.0,
        "mean_wer": sum(r["wer"] for r in usable_results) / len(usable_results) if usable_results else 1.0,
        "diacritic_preservation_rate": (
            sum(r["diacritic_preservation_rate"] for r in vi_results) / len(vi_results) if vi_results else 1.0
        ),
        "quote_containment_rate": (
            sum(1 for r in usable_results if r["quote_contained"]) / len(usable_results) if usable_results else 0.0
        ),
        "cer_by_language": {
            "vi": sum(r["cer"] for r in vi_results) / len(vi_results) if vi_results else None,
            "en": sum(r["cer"] for r in en_results) / len(en_results) if en_results else None,
        },
        "p50_seconds": percentile([r["seconds"] for r in results], 0.5),
        "p95_seconds": percentile([r["seconds"] for r in results], 0.95),
        "total_seconds": total_elapsed,
    }

    report = {
        "benchmark": "ocr_extraction_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "cer_threshold": cer_threshold,
        "ocr_provider": getattr(ocr_provider, "name", "cached_transcript"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "summary": summary,
        "details": results,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return report
