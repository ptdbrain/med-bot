from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.ingestion.audio_processor import AudioProcessor
from app.utils.file_utils import ALLOWED_AUDIO_EXTENSIONS, validate_extension, validate_size

router = APIRouter(prefix="/voice", tags=["voice"])


@router.post("/transcribe")
async def transcribe_voice(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        validate_size(content, request.app.state.rag_service.settings.max_upload_bytes)
        validate_extension(file.filename or "", ALLOWED_AUDIO_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    processor = AudioProcessor()
    transcript = processor.stt.transcribe(file.filename or "audio.wav", content)
    return {"transcript": transcript, "provider": "mock"}


@router.post("/chat")
async def voice_chat(request: Request, file: UploadFile = File(...), rag_mode: str = "naive") -> dict:
    content = await file.read()
    try:
        validate_size(content, request.app.state.rag_service.settings.max_upload_bytes)
        validate_extension(file.filename or "", ALLOWED_AUDIO_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    transcript = AudioProcessor().stt.transcribe(file.filename or "audio.wav", content)
    result = request.app.state.rag_service.chat(transcript, rag_mode=rag_mode)
    payload = result.__dict__
    payload["transcript"] = transcript
    return payload


@router.post("/ingest")
async def ingest_audio_lecture(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        validate_size(content, request.app.state.rag_service.settings.max_upload_bytes)
        validate_extension(file.filename or "", ALLOWED_AUDIO_EXTENSIONS)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audio_dir = request.app.state.rag_service.settings.data_dir / "raw" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    path = audio_dir / Path(file.filename or "audio.wav").name
    path.write_bytes(content)
    document = AudioProcessor().load(path, content)
    result = request.app.state.rag_service.pipeline.ingest_document(document)
    return result.__dict__
