from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.ingestion.image_processor import ImageProcessor
from app.ingestion.pipeline import IngestResult, IngestionPipeline
from app.llm.mock_llm import MockLLM
from app.rag.naive_rag import NaiveRAG, RagOutput
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.vector_store import InMemoryVectorStore
from app.schemas.document import SearchResult
from app.safety.medical_guardrails import MedicalGuardrails
from app.safety.pii_redaction import redact_pii
from app.utils.file_utils import ALLOWED_IMAGE_EXTENSIONS, ensure_directory, validate_extension, validate_size


SUPPORTED_RAG_MODES = {"naive"}


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict]
    retrieved_chunks: list[dict]
    mode_used: str
    safety_warning: str | None = None


class RAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        ensure_directory(self.settings.data_dir / "indexes")
        self.vector_store = InMemoryVectorStore(self.settings.index_path)
        self.metadata_store = SQLiteMetadataStore(self.settings.metadata_db_path)
        self.pipeline = IngestionPipeline(self.settings, self.vector_store, self.metadata_store)
        self.guardrails = MedicalGuardrails()
        self.llm = MockLLM()
        self.naive_rag = NaiveRAG(self.vector_store, self.llm, self.guardrails)

    def ingest_file(self, filename: str, content: bytes) -> IngestResult:
        return self.pipeline.ingest_file(filename, content)

    def ingest_image(self, filename: str, content: bytes) -> IngestResult:
        validate_size(content, self.settings.max_upload_bytes)
        validate_extension(filename, ALLOWED_IMAGE_EXTENSIONS)
        image_dir = ensure_directory(self.settings.data_dir / "raw" / "images")
        path = image_dir / Path(filename).name
        path.write_bytes(content)
        document = ImageProcessor().load(path, content)
        return self.pipeline.ingest_document(document)

    def chat(self, question: str, rag_mode: str = "naive", top_k: int = 5) -> ChatResult:
        clean_question = redact_pii(question)
        mode = rag_mode.strip().lower() or "naive"
        if mode != "naive":
            safety = self.guardrails.check(clean_question)
            if not safety.allowed:
                return self._format_output(RagOutput(safety.message or "", [], safety.safety_warning), mode)
            message = (
                f"RAG mode '{mode}' chưa được triển khai trong Phase 1. "
                "Hiện MVP hỗ trợ mode 'naive' để tránh tạo chức năng nửa vời."
            )
            return ChatResult(answer=self.guardrails.with_disclaimer(message), sources=[], retrieved_chunks=[], mode_used=mode)
        output = self.naive_rag.answer(clean_question, top_k=top_k)
        return self._format_output(output, "naive")

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return self.vector_store.search(redact_pii(query), top_k=top_k)

    def _format_output(self, output: RagOutput, mode: str) -> ChatResult:
        return ChatResult(
            answer=output.answer,
            sources=[self._source_dict(result) for result in output.results],
            retrieved_chunks=[result.to_dict() for result in output.results],
            mode_used=mode,
            safety_warning=output.safety_warning,
        )

    def _source_dict(self, result: SearchResult) -> dict:
        metadata = result.chunk.metadata
        return {
            "chunk_id": result.chunk.chunk_id,
            "title": metadata.title,
            "source_path": metadata.source_path,
            "source_type": metadata.source_type,
            "score": result.score,
            "page_number": metadata.page_number,
            "section_title": metadata.section_title,
        }
