from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class PdfLoader(DocumentLoader):
    source_type = "pdf"

    def load(self, path: Path, content: bytes) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to ingest PDF files.") from exc

        reader = PdfReader(BytesIO(content))
        pages: list[str] = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            pages.append(f"\n\n<!-- page:{index} -->\n{page_text}")
        return self._document(path, "\n".join(pages), {"page_count": len(reader.pages)})
