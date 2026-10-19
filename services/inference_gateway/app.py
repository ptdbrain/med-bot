from __future__ import annotations

import base64
import secrets
import uuid
from dataclasses import asdict
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.embeddings.factory import create_embedding_provider
from app.ingestion.ocr.factory import create_ocr_provider
from app.llm.factory import create_llm
from app.retrieval.index_manifest import embedding_fingerprint
from app.retrieval.reranker_factory import create_reranker
from app.schemas.document import Chunk, ChunkMetadata, SearchResult
from app.schemas.answer import CitedClaim
from app.rag.context_builder import PackedContext

Text = Annotated[str, Field(min_length=1, max_length=24000)]


class EmbedRequest(BaseModel):
    texts: list[Text] = Field(min_length=1, max_length=64)
    purpose: Literal["query", "document"]


class OCRRequest(BaseModel):
    image_base64: str = Field(max_length=32_000_000)
    page_number: int = Field(ge=1)
    languages: list[str] = Field(default=["vi", "en"], max_length=4)


class Candidate(BaseModel):
    id: str
    text: Text


class RerankRequest(BaseModel):
    query: Text
    documents: list[Candidate] = Field(min_length=1, max_length=80)


class GenerateRequest(BaseModel):
    question: Text
    sources: list[dict] = Field(max_length=20)
    max_output_tokens: int = Field(default=1024, ge=1, le=2048)


class VerifyRequest(BaseModel):
    claims: list[CitedClaim] = Field(max_length=12)
    source_map: dict[str, dict]


def source_result(identifier, text, title="", page_number=None):
    return SearchResult(Chunk(identifier, text, ChunkMetadata("gateway", "gateway", "text", title,
        "", identifier, page_number=page_number)), 1.0)


def create_gateway(settings: Settings | None = None, *, embedding=None, ocr=None, llm=None, reranker=None, verifier=None):
    settings = settings or get_settings()
    if "http" in {settings.llm_provider, settings.embedding_provider, settings.reranker_provider, settings.ocr_provider}:
        raise ValueError("Gateway requires local providers; recursive HTTP profile is not permitted")
    embedding = embedding if embedding is not None else create_embedding_provider(settings)
    ocr = ocr if ocr is not None else create_ocr_provider(settings)
    llm, reranker = llm or create_llm(settings), reranker or create_reranker(settings)
    app = FastAPI(title="MedTutor inference gateway")
    bearer = HTTPBearer(auto_error=False)

    def authorize(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
        if not settings.http_provider_key:
            raise HTTPException(503, "Gateway credentials are not configured")
        if not credentials or not secrets.compare_digest(credentials.credentials, settings.http_provider_key):
            raise HTTPException(401, "Authentication required")

    @app.exception_handler(RuntimeError)
    def provider_error(request, exc):
        return JSONResponse({"detail": "Provider unavailable"}, status_code=503)

    def envelope(data):
        return {**data, "request_id": str(uuid.uuid4())}

    @app.get("/v1/health", dependencies=[Depends(authorize)])
    def health():
        return envelope({"embedding_fingerprint": embedding_fingerprint(settings)})

    @app.post("/v1/embeddings", dependencies=[Depends(authorize)])
    def embed(body: EmbedRequest):
        if embedding is None:
            raise HTTPException(503, "Embedding provider not configured")
        vectors = ([embedding.embed_query(text) for text in body.texts] if body.purpose == "query"
                   else embedding.embed_documents(body.texts))
        return envelope({"vectors": vectors, "model_fingerprint": embedding_fingerprint(settings)})

    @app.post("/v1/ocr", dependencies=[Depends(authorize)])
    def recognize(body: OCRRequest):
        if ocr is None:
            raise HTTPException(503, "OCR provider not configured")
        try:
            image = base64.b64decode(body.image_base64, validate=True)
        except ValueError as exc:
            raise HTTPException(422, "Invalid image encoding") from exc
        return envelope(asdict(ocr.recognize(image, page_number=body.page_number, languages=tuple(body.languages))))

    @app.post("/v1/rerank", dependencies=[Depends(authorize)])
    def rank(body: RerankRequest):
        candidates = [source_result(row.id, row.text) for row in body.documents]
        ranked = reranker.rerank(body.query, candidates, top_k=len(candidates))
        return envelope({"score_type": "normalized", "results": [{"id": row.chunk.chunk_id, "score": row.score} for row in ranked]})

    @app.post("/v1/generate", dependencies=[Depends(authorize)])
    def generate(body: GenerateRequest):
        if sum(len(str(source)) for source in body.sources) > 100_000:
            raise HTTPException(413, "Source context too large")
        results = [source_result(str(i), str(row.get("text", "")), str(row.get("title", "")), row.get("page_number")) for i, row in enumerate(body.sources, 1)]
        if hasattr(llm, "generate_result"):
            result = llm.generate_result(body.question, results)
            return envelope({"draft": result.draft.model_dump() if result.draft else None,
                "raw_text": result.raw_text, "finish_reason": result.finish_reason, "usage": result.usage})
        return envelope({"raw_text": llm.generate(body.question, results), "draft": None, "finish_reason": "stop"})

    @app.post("/v1/verify", dependencies=[Depends(authorize)])
    def verify(body: VerifyRequest):
        provider = verifier
        if provider is None and settings.llm_provider == "ollama":
            from app.llm.ollama_verifier import OllamaSupportVerifier
            provider = OllamaSupportVerifier(settings)
        if provider is None:
            raise HTTPException(503, "Support verifier unavailable")
        packed = PackedContext([], 0, body.source_map, [])
        return envelope({"results": [asdict(r) for r in provider.verify(body.claims, packed)]})
    return app


app = create_gateway()
