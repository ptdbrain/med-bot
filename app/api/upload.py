from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/upload", tags=["upload"])


@router.get("/limits")
def upload_limits() -> dict:
    return {
        "document_extensions": [".txt", ".md", ".pdf", ".docx", ".html", ".htm"],
        "image_extensions": [".png", ".jpg", ".jpeg"],
        "audio_extensions": [".wav", ".mp3", ".m4a", ".ogg"],
        "video_extensions": [".mp4", ".mov", ".mkv", ".webm"],
        "max_upload_mb": 15,
    }
