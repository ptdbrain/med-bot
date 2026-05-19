from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.ingestion.chunker import TextChunker
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.html_loader import HtmlLoader
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.text_loader import TextLoader
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.document import Chunk, Document
from app.utils.file_utils import ALLOWED_DOCUMENT_EXTENSIONS, ensure_directory, validate_extension, validate_size


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    title: str
    chunks_added: int
    source_path: str
    source_type: str


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        vector_store: InMemoryVectorStore,
        metadata_store: SQLiteMetadataStore,
        chunker: TextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.chunker = chunker or TextChunker()

    def ingest_file(self, filename: str, content: bytes) -> IngestResult:
        validate_size(content, self.settings.max_upload_bytes)
        extension = validate_extension(filename, ALLOWED_DOCUMENT_EXTENSIONS)
        path = self._write_raw_file(filename, content)
        document = self._load_document(path, content, extension)
        chunks = self._store_document(document)
        return IngestResult(
            doc_id=document.doc_id,
            title=document.title,
            chunks_added=len(chunks),
            source_path=document.source_path,
            source_type=document.source_type,
        )

    def ingest_document(self, document: Document) -> IngestResult:
        chunks = self._store_document(document)
        return IngestResult(
            doc_id=document.doc_id,
            title=document.title,
            chunks_added=len(chunks),
            source_path=document.source_path,
            source_type=document.source_type,
        )

    def _write_raw_file(self, filename: str, content: bytes) -> Path:
        safe_name = Path(filename).name
        raw_dir = ensure_directory(self.settings.data_dir / "raw")
        path = raw_dir / safe_name
        path.write_bytes(content)
        return path

    def _load_document(self, path: Path, content: bytes, extension: str) -> Document:
        if extension in {".txt", ".md"}:
            return TextLoader().load(path, content)
        if extension in {".html", ".htm"}:
            return HtmlLoader().load(path, content)
        if extension == ".pdf":
            return PdfLoader().load(path, content)
        if extension == ".docx":
            return DocxLoader().load(path, content)
        raise ValueError(f"Unsupported extension: {extension}")

    def _store_document(self, document: Document) -> list[Chunk]:
        chunks = self.chunker.chunk(document)
        self.metadata_store.upsert_document(document)
        self.metadata_store.upsert_chunks(chunks)
        self.vector_store.add_chunks(chunks)
        return chunks
