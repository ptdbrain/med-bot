from __future__ import annotations

from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class TextLoader(DocumentLoader):
    source_type = "text"

    def load(self, path: Path, content: bytes) -> Document:
        text = content.decode("utf-8", errors="replace")
        suffix = path.suffix.lower().lstrip(".") or "txt"
        self.source_type = suffix
        return self._document(path, text)
