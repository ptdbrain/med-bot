from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()
        if cleaned:
            self.parts.append(cleaned)


class HtmlLoader(DocumentLoader):
    source_type = "html"

    def load(self, path: Path, content: bytes) -> Document:
        parser = _TextExtractor()
        parser.feed(content.decode("utf-8", errors="replace"))
        return self._document(path, "\n".join(parser.parts))
