from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.schemas.document import Document
from app.utils.hashing import short_hash


class DocumentLoader(ABC):
    source_type: str

    @abstractmethod
    def load(self, path: Path, content: bytes) -> Document:
        raise NotImplementedError

    def _document(self, path: Path, text: str, metadata: dict | None = None) -> Document:
        title = path.stem.replace("_", " ").replace("-", " ").strip() or path.name
        doc_id = f"{self.source_type}-{short_hash(path.as_posix() + text)}"
        return Document(
            doc_id=doc_id,
            title=title,
            source_path=path.as_posix(),
            source_type=self.source_type,
            text=text,
            metadata=metadata or {},
        )
