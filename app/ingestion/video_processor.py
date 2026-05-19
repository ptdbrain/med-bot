from __future__ import annotations

from pathlib import Path

from app.ingestion.audio_processor import MockSpeechToText
from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class VideoProcessor(DocumentLoader):
    source_type = "video"

    def __init__(self, stt: MockSpeechToText | None = None) -> None:
        self.stt = stt or MockSpeechToText()

    def load(self, path: Path, content: bytes) -> Document:
        transcript = self.stt.transcribe(path.name, content)
        text = "Video transcript MVP:\n" + transcript
        return self._document(path, text, {"transcript_provider": "mock", "keyframes": []})
