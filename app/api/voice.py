from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter(prefix="/voice", tags=["voice"])


def _raise_audio_error(exc: Exception) -> None:
    status_code = 503 if isinstance(exc, RuntimeError) else 400
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.post("/transcribe")
async def transcribe_voice(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        transcription = request.app.state.rag_service.transcribe_audio(
            file.filename or "audio.wav",
            content,
        )
    except (RuntimeError, ValueError) as exc:
        _raise_audio_error(exc)
    payload = transcription.to_dict()
    payload["transcript"] = transcription.text
    return payload


@router.post("/chat")
async def voice_chat(
    request: Request,
    file: UploadFile = File(...),
    rag_mode: str = "naive",
    top_k: int = 5,
) -> dict:
    content = await file.read()
    try:
        transcription, result = request.app.state.rag_service.voice_chat(
            file.filename or "audio.wav",
            content,
            rag_mode=rag_mode,
            top_k=top_k,
        )
    except (RuntimeError, ValueError) as exc:
        _raise_audio_error(exc)
    payload = dict(result.__dict__)
    payload["transcript"] = transcription.text
    payload["transcription"] = transcription.to_dict()
    return payload


@router.post("/ingest")
async def ingest_audio_lecture(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        result, transcription = request.app.state.rag_service.ingest_audio(
            file.filename or "audio.wav",
            content,
        )
    except (RuntimeError, ValueError) as exc:
        _raise_audio_error(exc)
    payload = dict(result.__dict__)
    payload["transcription"] = transcription.to_dict()
    return payload
