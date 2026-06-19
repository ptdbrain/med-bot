from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile


router = APIRouter(prefix="/video", tags=["video"])


@router.post("/ingest")
async def ingest_video(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        result, transcription = request.app.state.rag_service.ingest_video(
            file.filename or "video.mp4",
            content,
        )
    except (RuntimeError, ValueError) as exc:
        status_code = 503 if isinstance(exc, RuntimeError) else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    payload = dict(result.__dict__)
    payload["transcription"] = transcription.to_dict()
    return payload
