from __future__ import annotations

from pathlib import Path

ALLOWED_DOCUMENT_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_extension(filename: str, allowed: set[str]) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in allowed:
        allowed_list = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported file extension '{extension}'. Allowed: {allowed_list}")
    return extension


def validate_size(content: bytes, max_bytes: int) -> None:
    if len(content) > max_bytes:
        raise ValueError(f"File is too large. Limit is {max_bytes} bytes.")
