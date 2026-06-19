from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat, documents, eval, health, image, ingest, upload, video, voice
from app.config import Settings, get_settings
from app.rag.service import RAGService
from app.utils.logging import configure_logging


def create_app(settings: Settings | None = None) -> FastAPI:
    configure_logging()
    settings = settings or get_settings()
    application = FastAPI(
        title=settings.project_name,
        description="Medical education RAG copilot with citations and safety guardrails.",
        version="0.1.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.frontend_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.rag_service = RAGService(settings)
    application.include_router(health.router)
    application.include_router(ingest.router)
    application.include_router(upload.router)
    application.include_router(documents.router)
    application.include_router(chat.router)
    application.include_router(voice.router)
    application.include_router(video.router)
    application.include_router(image.router)
    application.include_router(eval.router)
    return application


app = create_app()
