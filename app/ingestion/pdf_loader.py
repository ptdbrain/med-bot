from __future__ import annotations

import uuid
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.ingestion.extraction_quality import assess_text, classify_page
from app.ingestion.layout import order_blocks
from app.ingestion.normalizer import normalize_pages, split_text_blocks
from app.ingestion.ocr.paddle import OCRWorkerError
from app.ingestion.pdf_renderer import PDFRenderer
from app.schemas.document import Document
from app.schemas.extraction import ExtractionReport, PageExtraction


class PdfLoader(DocumentLoader):
    source_type = "pdf"

    def __init__(
        self,
        ocr=None,
        renderer=None,
        *,
        dpi=250,
        max_pages=1000,
        languages=("vi", "en"),
        correlation_id=None,
        job_id=None,
    ):
        self.ocr, self.renderer = ocr, renderer or PDFRenderer()
        self.dpi, self.max_pages, self.languages = dpi, max_pages, languages
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.job_id = job_id

    def extract_page(self, content, number, text, image_coverage, has_visual_content, native_blocks=None):
        classification = classify_page(text, image_coverage=image_coverage, has_visual_content=has_visual_content)
        page = PageExtraction(
            number, classification, native_blocks or split_text_blocks(text, number), metrics=assess_text(text)
        )
        if classification == "blank":
            page.status = "blank"
        elif classification in {"scan", "mixed"}:
            if self.ocr is None:
                page.status = "failed" if classification == "scan" else "review"
                page.warnings.append("ocr_unavailable")
                return page
            try:
                image = self.renderer.render_page(content, number, self.dpi)
                result = self.ocr.recognize(
                    image,
                    page_number=number,
                    languages=self.languages,
                    correlation_id=self.correlation_id,
                    job_id=self.job_id,
                    dpi=self.dpi,
                )
                page.warnings.extend(result.warnings)
                page.diagnostics.update(
                    {
                        "correlation_id": self.correlation_id,
                        "job_id": self.job_id,
                        "page_number": number,
                        "dpi": self.dpi,
                    }
                )
                page.diagnostics.update(result.diagnostics)
                known = {" ".join(b.text.casefold().split()) for b in page.blocks}
                page.blocks.extend(b for b in result.blocks if " ".join(b.text.casefold().split()) not in known)
                if result.confidence is not None:
                    page.metrics["ocr_confidence"] = result.confidence
                    if result.confidence < 0.5:
                        page.status = "review"
                if not result.blocks:
                    page.status = "failed" if classification == "scan" else "review"
                    page.warnings.append("empty_ocr")
            except OCRWorkerError as exc:
                page.status = "failed" if classification == "scan" else "review"
                page.warnings.append("ocr_failed")
                page.diagnostics.update(exc.diagnostics)
            except (RuntimeError, ImportError, ValueError, OSError) as exc:
                page.status = "failed" if classification == "scan" else "review"
                page.warnings.append("ocr_failed")
                page.diagnostics.update(
                    {
                        "correlation_id": self.correlation_id,
                        "job_id": self.job_id,
                        "page_number": number,
                        "dpi": self.dpi,
                        "failure_category": "ocr_provider_error",
                        "error_type": type(exc).__name__,
                    }
                )
        page.blocks = order_blocks(page.blocks)
        return page

    def load(self, path: Path, content: bytes) -> Document:
        if not content.startswith(b"%PDF-"):
            raise ValueError("Invalid PDF signature")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to ingest PDF files.") from exc

        reader = PdfReader(BytesIO(content))
        if reader.is_encrypted:
            raise ValueError("Encrypted PDF is not supported")
        if not 0 < len(reader.pages) <= self.max_pages:
            raise ValueError("PDF page limit exceeded or empty PDF")
        pages = []
        session_factory = getattr(self.ocr, "document_session", None)
        session = session_factory() if session_factory else nullcontext()
        with session:
            for number, source in enumerate(reader.pages, 1):
                try:
                    text = source.extract_text() or ""
                    try:
                        blocks, coverage, visual = self.renderer.inspect_page(content, number)
                    except ImportError:
                        blocks, coverage = None, 1.0 if "/XObject" in source.get("/Resources", {}) else 0.0
                        visual = bool(text or coverage)
                    page = self.extract_page(content, number, text, coverage, visual, blocks)
                except (ValueError, RuntimeError, OSError):
                    page = PageExtraction(number, "mixed", warnings=["page_extraction_failed"], status="failed")
                pages.append(page)
        pages = normalize_pages(pages)
        report = ExtractionReport(pages, correlation_id=self.correlation_id, job_id=self.job_id)
        text = "\n\n".join(
            f"<!-- page:{p.page_number} -->\n" + "\n\n".join(b.text for b in p.blocks)
            for p in pages
            if p.status != "failed"
        )
        if not any(p.blocks and p.status != "failed" for p in pages):
            failures = "; ".join(
                f"page {page.page_number}: {','.join(page.warnings) or page.status}"
                for page in pages
                if page.status == "failed"
            )
            suffix = f" ({failures})" if failures else ""
            raise PDFExtractionError("PDF contains no usable text; configure OCR for scanned pages" + suffix, report)
        return self._document(
            path,
            text,
            {
                "page_count": len(pages),
                "extraction": report.to_dict(),
                "requires_review": any(p.status in {"review", "failed"} for p in pages),
            },
        )


class PDFExtractionError(ValueError):
    def __init__(self, message: str, report: ExtractionReport | dict):
        super().__init__(message)
        self.report = report.to_dict() if isinstance(report, ExtractionReport) else report
