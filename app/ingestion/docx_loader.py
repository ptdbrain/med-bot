from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class DocxLoader(DocumentLoader):
    source_type = "docx"

    def load(self, path: Path, content: bytes) -> Document:
        try:
            from docx import Document as DocxDocument
        except ImportError as exc:
            raise RuntimeError("Install python-docx to ingest DOCX files.") from exc

        document = DocxDocument(BytesIO(content))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
        return self._document(path, text)
