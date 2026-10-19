from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings
from app.ingestion.chunk_quality import assess_chunks
from app.ingestion.chunker import TextChunker
from app.ingestion.docx_loader import DocxLoader
from app.ingestion.html_loader import HtmlLoader
from app.ingestion.identity import document_id
from app.ingestion.ocr.factory import create_ocr_provider
from app.ingestion.pdf_loader import PdfLoader
from app.ingestion.pdf_renderer import PDFRenderer
from app.ingestion.text_loader import TextLoader
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.publication import IndexPublisher
from app.retrieval.vector_store import VectorStore
from app.schemas.document import Chunk, Document
from app.utils.file_utils import ALLOWED_DOCUMENT_EXTENSIONS, ensure_directory, validate_extension, validate_size


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    title: str
    chunks_added: int
    source_path: str
    source_type: str
    revision_id: str | None = None
    report: dict = field(default_factory=dict)


class IngestReviewRequired(ValueError):
    def __init__(self, report):
        super().__init__("Document requires extraction review before publication")
        self.report = report


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        metadata_store: SQLiteMetadataStore,
        chunker: TextChunker | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.ocr_provider = create_ocr_provider(settings)
        self.publisher = IndexPublisher(metadata_store, vector_store)
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_size=settings.chunk_min_size,
        )
        if chunker is None and settings.chunk_unit == "tokens":
            from app.ingestion.token_chunker import TokenChunker
            from app.ingestion.tokenizer import ModelTokenCounter

            self.chunker = TokenChunker(
                ModelTokenCounter(settings.embedding_model, settings.embedding_revision),
                settings.chunk_target_tokens,
                settings.chunk_overlap_tokens,
                settings.chunk_min_tokens,
                settings.chunk_max_tokens,
            )

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        *,
        replace_doc_id: str | None = None,
        allow_partial: bool = False,
        progress=None,
    ) -> IngestResult:
        validate_size(content, self.settings.max_upload_bytes)
        extension = validate_extension(filename, ALLOWED_DOCUMENT_EXTENSIONS)
        safe_name = Path(filename.replace("\\", "/")).name
        digest = hashlib.sha256(content).hexdigest()
        doc_id = replace_doc_id or document_id("uploads", safe_name + ":" + digest)
        if replace_doc_id and self.metadata_store.get_document(replace_doc_id) is None:
            raise ValueError("Replacement document does not exist")
        path = self.settings.data_dir / "raw" / doc_id / digest / safe_name
        document = self._load_document(path, content, extension)
        document.doc_id = doc_id
        document.title = Path(safe_name).stem.replace("_", " ")
        document.metadata["content_hash"] = digest
        document.metadata["original_filename"] = safe_name
        if document.metadata.get("requires_review") and not allow_partial:
            raise IngestReviewRequired(document.metadata.get("extraction", {}))
        if not document.text.strip():
            raise ValueError("Document contains no usable text")
        ensure_directory(path.parent)
        if not path.exists():
            path.write_bytes(content)
        if progress:
            progress("chunking", document.metadata.get("extraction", {}))
        chunks = self._store_document(document, progress=progress)
        return IngestResult(
            doc_id=document.doc_id,
            title=document.title,
            chunks_added=len(chunks),
            source_path=document.source_path,
            source_type=document.source_type,
            revision_id=document.metadata.get("revision_id"),
            report={"extraction": document.metadata.get("extraction", {}), "chunk_quality": assess_chunks(chunks)},
        )

    def ingest_document(self, document: Document) -> IngestResult:
        chunks = self._store_document(document)
        return IngestResult(
            doc_id=document.doc_id,
            title=document.title,
            chunks_added=len(chunks),
            source_path=document.source_path,
            source_type=document.source_type,
            revision_id=document.metadata.get("revision_id"),
            report={"extraction": document.metadata.get("extraction", {}), "chunk_quality": assess_chunks(chunks)},
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
            return PdfLoader(
                ocr=self.ocr_provider,
                renderer=PDFRenderer(self.settings.pdf_max_pixels),
                dpi=self.settings.ocr_dpi,
                max_pages=self.settings.pdf_max_pages,
                languages=self.settings.ocr_languages,
            ).load(path, content)
        if extension == ".docx":
            return DocxLoader().load(path, content)
        raise ValueError(f"Unsupported extension: {extension}")

    def _store_document(self, document: Document, progress=None) -> list[Chunk]:
        chunks = self.chunker.chunk(document)
        if not chunks:
            raise ValueError("Document contains no usable chunks")
        if "parents" not in document.metadata:
            sections = TextChunker()._split_sections(document.text)
            document.metadata["parents"] = {
                chunk.metadata.parent_id: {"text": text, "page_number": page, "section_title": title}
                for title, page, text in sections
                for chunk in chunks
                if chunk.metadata.section_title == title and chunk.metadata.page_number == page
            }
        previous = self.publisher.active_revision(document.doc_id)
        if progress:
            progress("embedding", assess_chunks(chunks))
        revision = self.publisher.stage(document, chunks, report=assess_chunks(chunks))
        if progress:
            progress("publishing", {"revision_id": revision})
        self.publisher.publish(document.doc_id, revision, previous)
        return chunks
