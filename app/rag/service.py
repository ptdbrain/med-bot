from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.embeddings.base import EmbeddingProvider
from app.embeddings.factory import create_embedding_provider
from app.ingestion.audio_processor import (
    AudioProcessor,
    SpeechToTextProvider,
    TranscriptionResult,
    create_speech_to_text,
)
from app.ingestion.html_loader import HtmlLoader
from app.ingestion.image_processor import ImageProcessor
from app.ingestion.ocr.factory import create_ocr_provider
from app.ingestion.pipeline import IngestionPipeline, IngestResult
from app.ingestion.video_processor import VideoProcessor
from app.llm.base import LLMProvider
from app.llm.factory import create_llm
from app.rag.agentic_router import RuleBasedAgenticRouter
from app.rag.answer_service import AnswerService
from app.rag.context_builder import ContextBuilder
from app.rag.corrective_rag import CorrectiveRAG
from app.rag.hybrid_rag import HybridRAG
from app.rag.multi_query_rag import MultiQueryRAG
from app.rag.naive_rag import NaiveRAG, RagOutput
from app.rag.parent_child_rag import ParentChildRAG
from app.rag.rerank_rag import RerankRAG
from app.retrieval.bm25_store import BM25Store
from app.retrieval.factory import create_vector_store
from app.retrieval.metadata_store import SQLiteMetadataStore
from app.retrieval.publication import ActiveVectorStore
from app.retrieval.query_transform import RuleBasedQueryTransformer
from app.retrieval.relevance import RelevancePolicy, filter_relevant_results
from app.retrieval.reranker_factory import create_reranker
from app.retrieval.vector_store import VectorStore
from app.safety.medical_guardrails import MedicalGuardrails
from app.safety.pii_redaction import redact_pii
from app.schemas.document import SearchResult
from app.utils.file_utils import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ensure_directory,
    validate_extension,
    validate_size,
)
from app.utils.hashing import short_hash

SUPPORTED_RAG_MODES = {
    "naive",
    "hybrid",
    "rerank",
    "parent_child",
    "multi_query",
    "corrective",
    "agentic",
    "lexical_vector",
    "lexical_hybrid",
    "lexical_rerank",
}


@dataclass(frozen=True)
class ChatResult:
    answer: str
    sources: list[dict]
    retrieved_chunks: list[dict]
    mode_used: str
    safety_warning: str | None = None
    validation: dict | None = None


class RAGService:
    def __init__(
        self,
        settings: Settings | None = None,
        stt_provider: SpeechToTextProvider | None = None,
        llm: LLMProvider | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        ensure_directory(self.settings.data_dir / "indexes")
        self.metadata_store = SQLiteMetadataStore(self.settings.metadata_db_path)
        self.embedding_provider = (
            embedding_provider if embedding_provider is not None else create_embedding_provider(self.settings)
        )
        self.vector_store = (
            vector_store if vector_store is not None else create_vector_store(self.settings, self.embedding_provider)
        )
        self.vector_store = ActiveVectorStore(self.vector_store, self.metadata_store)
        self.pipeline = IngestionPipeline(self.settings, self.vector_store, self.metadata_store)
        self.guardrails = MedicalGuardrails()
        self.llm = llm if llm is not None else create_llm(self.settings)
        self.relevance_policy = RelevancePolicy(
            self.settings.relevance_dense_threshold, self.settings.relevance_reranker_threshold
        )
        self.context_builder = ContextBuilder()
        verifier = None
        if self.settings.llm_provider == "ollama" and self.settings.answer_validation == "strict":
            from app.llm.ollama_verifier import OllamaSupportVerifier

            verifier = OllamaSupportVerifier(self.settings)
        elif self.settings.llm_provider == "http" and self.settings.answer_validation == "strict":
            from app.llm.http_llm import HTTPSupportVerifier

            verifier = HTTPSupportVerifier(self.settings)
        self.answer_service = AnswerService(
            self.llm, self.guardrails, strict=self.settings.answer_validation == "strict", verifier=verifier
        )
        speech_provider = stt_provider or create_speech_to_text(self.settings)
        self.audio_processor = AudioProcessor(speech_provider)
        self.image_processor = ImageProcessor(ocr=create_ocr_provider(self.settings))
        self.video_processor = VideoProcessor(speech_provider)
        self.bm25_store = BM25Store()
        self.bm25_store.add_chunks(self.metadata_store.list_chunks())
        self._bm25_epoch = self.metadata_store.epoch
        self.naive_rag = NaiveRAG(self.vector_store, self.llm, self.guardrails)
        self.hybrid_rag = HybridRAG(
            self.vector_store,
            self.bm25_store,
            self.llm,
            self.guardrails,
            candidate_multiplier=self.settings.hybrid_candidate_multiplier,
            rrf_k=self.settings.hybrid_rrf_k,
            dense_weight=self.settings.hybrid_dense_weight,
            sparse_weight=self.settings.hybrid_sparse_weight,
            dense_k=self.settings.dense_candidate_k,
            sparse_k=self.settings.sparse_candidate_k,
        )
        self.query_transformer = RuleBasedQueryTransformer()
        self.rerank_rag = RerankRAG(
            self.hybrid_rag,
            create_reranker(self.settings),
            self.llm,
            self.guardrails,
            candidate_k=self.settings.reranker_candidate_k,
        )
        self.parent_child_rag = ParentChildRAG(
            self.hybrid_rag,
            self.metadata_store,
            self.llm,
            self.guardrails,
        )
        self.multi_query_rag = MultiQueryRAG(
            self.hybrid_rag,
            self.query_transformer,
            self.llm,
            self.guardrails,
        )
        self.corrective_rag = CorrectiveRAG(
            self.hybrid_rag,
            self.query_transformer,
            self.llm,
            self.guardrails,
        )
        self.agentic_router = RuleBasedAgenticRouter()

    @property
    def indexed_chunk_count(self) -> int:
        return self.metadata_store.count_chunks()

    @property
    def vector_indexed_count(self) -> int | None:
        count = getattr(self.vector_store, "count", None)
        return int(count) if count is not None else None

    def ingest_file(
        self,
        filename: str,
        content: bytes,
        *,
        replace_doc_id: str | None = None,
        allow_partial: bool = False,
        progress=None,
        correlation_id: str | None = None,
        job_id: str | None = None,
    ) -> IngestResult:
        result = self.pipeline.ingest_file(
            filename,
            content,
            replace_doc_id=replace_doc_id,
            allow_partial=allow_partial,
            progress=progress,
            correlation_id=correlation_id,
            job_id=job_id,
        )
        self._sync_bm25()
        return result

    def ingest_image(self, filename: str, content: bytes) -> IngestResult:
        validate_size(content, self.settings.max_upload_bytes)
        validate_extension(filename, ALLOWED_IMAGE_EXTENSIONS)
        image_dir = ensure_directory(self.settings.data_dir / "raw" / "images")
        path = image_dir / Path(filename).name
        path.write_bytes(content)
        document = self.image_processor.load(path, content)
        result = self.pipeline.ingest_document(document)
        self._sync_bm25()
        return result

    def ingest_video(
        self,
        filename: str,
        content: bytes,
    ) -> tuple[IngestResult, TranscriptionResult]:
        validate_size(content, self.settings.max_upload_bytes)
        safe_name = Path(filename).name
        validate_extension(safe_name, ALLOWED_VIDEO_EXTENSIONS)
        video_dir = ensure_directory(self.settings.data_dir / "raw" / "video")
        path = video_dir / safe_name
        processed = self.video_processor.process(path, content)
        path.write_bytes(content)
        result = self.pipeline.ingest_document(processed.document)
        self._sync_bm25()
        return result, processed.transcription

    def ingest_url(self, url: str, content: bytes) -> IngestResult:
        validate_size(content, self.settings.max_upload_bytes)
        filename = f"url-{short_hash(url)}.html"
        path = self.settings.data_dir / "raw" / filename
        document = HtmlLoader().load(path, content)
        document.metadata["url"] = url
        result = self.pipeline.ingest_document(document)
        self._sync_bm25()
        return result

    def list_documents(self) -> list[dict]:
        return self.metadata_store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        document = self.metadata_store.get_document(doc_id)
        if document is None:
            return False
        chunk_ids = self.metadata_store.chunk_ids_for_document(doc_id)
        if chunk_ids:
            self.vector_store.delete_chunks(chunk_ids)
        deleted = self.metadata_store.delete_document(doc_id)
        if deleted:
            self._delete_raw_source(document["source_path"])
        self._sync_bm25()
        return deleted

    def transcribe_audio(self, filename: str, content: bytes) -> TranscriptionResult:
        safe_name = self._validate_audio(filename, content)
        return self.audio_processor.transcribe(safe_name, content)

    def voice_chat(
        self,
        filename: str,
        content: bytes,
        rag_mode: str = "naive",
        top_k: int = 5,
    ) -> tuple[TranscriptionResult, ChatResult]:
        transcription = self.transcribe_audio(filename, content)
        chat_result = self.chat(transcription.text, rag_mode=rag_mode, top_k=top_k)
        return transcription, chat_result

    def ingest_audio(
        self,
        filename: str,
        content: bytes,
    ) -> tuple[IngestResult, TranscriptionResult]:
        safe_name = self._validate_audio(filename, content)
        audio_dir = ensure_directory(self.settings.data_dir / "raw" / "audio")
        path = audio_dir / safe_name
        processed = self.audio_processor.process(path, content)
        path.write_bytes(content)
        result = self.pipeline.ingest_document(processed.document)
        self._sync_bm25()
        return result, processed.transcription

    def chat(self, question: str, rag_mode: str = "naive", top_k: int = 5) -> ChatResult:
        clean_question = redact_pii(question)
        mode = rag_mode.strip().lower() or "naive"
        safety = self.guardrails.check(clean_question)
        if not safety.allowed:
            return self._format_output(
                RagOutput(
                    safety.message or "",
                    [],
                    safety.safety_warning,
                ),
                mode,
            )
        if mode not in SUPPORTED_RAG_MODES:
            message = (
                f"RAG mode '{mode}' chưa được triển khai. "
                f"Hiện hệ thống hỗ trợ: {', '.join(sorted(SUPPORTED_RAG_MODES))}."
            )
            return ChatResult(
                answer=self.guardrails.with_disclaimer(message), sources=[], retrieved_chunks=[], mode_used=mode
            )
        if mode == "agentic":
            selected_mode = self.agentic_router.choose_mode(clean_question)
        elif mode == "lexical_hybrid":
            selected_mode = "hybrid"
        elif mode == "lexical_rerank":
            selected_mode = "rerank"
        elif mode == "lexical_vector":
            selected_mode = "naive"
        else:
            selected_mode = mode
        self._ensure_bm25_current()
        results = filter_relevant_results(
            clean_question, self.retrieve(clean_question, top_k, selected_mode), self.relevance_policy
        )
        budget = self.settings.context_window_tokens - self.settings.context_output_reserve - 2048
        packed = self.context_builder.pack(clean_question, results, input_budget=max(0, budget))
        output = self.answer_service.answer(clean_question, packed)
        return self._format_output(output, mode)

    def retrieve(self, query: str, top_k: int = 5, rag_mode: str = "naive") -> list[SearchResult]:
        self._ensure_bm25_current()
        clean_query = redact_pii(query)
        mode = rag_mode.strip().lower() or "naive"
        if mode not in SUPPORTED_RAG_MODES:
            raise ValueError(
                f"RAG mode '{mode}' chưa được triển khai. "
                f"Hiện hệ thống hỗ trợ: {', '.join(sorted(SUPPORTED_RAG_MODES))}."
            )
        if mode == "agentic":
            mode = self.agentic_router.choose_mode(clean_query)
        if mode in {"naive", "lexical_vector"}:
            return self.vector_store.search(clean_query, top_k=top_k)
        if mode == "lexical_hybrid":
            return self._rag_for_mode("hybrid").retrieve(clean_query, top_k=top_k)
        if mode == "lexical_rerank":
            return self._rag_for_mode("rerank").retrieve(clean_query, top_k=top_k)
        return self._rag_for_mode(mode).retrieve(clean_query, top_k=top_k)

    def _sync_bm25(self) -> None:
        self.bm25_store.replace_chunks(self.metadata_store.list_chunks())
        self._bm25_epoch = self.metadata_store.epoch

    def _ensure_bm25_current(self) -> None:
        if self._bm25_epoch != self.metadata_store.epoch:
            self._sync_bm25()

    def _validate_audio(self, filename: str, content: bytes) -> str:
        validate_size(content, self.settings.max_upload_bytes)
        safe_name = Path(filename).name
        validate_extension(safe_name, ALLOWED_AUDIO_EXTENSIONS)
        return safe_name

    def _delete_raw_source(self, source_path: str) -> None:
        raw_root = (self.settings.data_dir / "raw").resolve()
        candidate = Path(source_path)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.is_relative_to(raw_root) and candidate.is_file():
            candidate.unlink()

    def _format_output(self, output: RagOutput, mode: str) -> ChatResult:
        return ChatResult(
            answer=output.answer,
            sources=[self._source_dict(result) for result in output.results],
            retrieved_chunks=[result.to_dict() for result in output.results],
            mode_used=mode,
            safety_warning=output.safety_warning,
            validation=output.validation,
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

    def _rag_for_mode(self, mode: str):
        pipelines = {
            "naive": self.naive_rag,
            "hybrid": self.hybrid_rag,
            "rerank": self.rerank_rag,
            "parent_child": self.parent_child_rag,
            "multi_query": self.multi_query_rag,
            "corrective": self.corrective_rag,
        }
        return pipelines[mode]
