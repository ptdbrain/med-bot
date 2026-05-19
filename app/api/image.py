from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

router = APIRouter(prefix="/image", tags=["image"])


@router.post("/ingest")
async def ingest_image(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        result = request.app.state.rag_service.ingest_image(file.filename or "image.png", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.__dict__
