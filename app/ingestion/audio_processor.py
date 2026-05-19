from __future__ import annotations

from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class MockSpeechToText:
    def transcribe(self, filename: str, content: bytes) -> str:
        return f"Mock transcript cho {filename}. Cau hinh Whisper/faster-whisper de dung am thanh that."


class AudioProcessor(DocumentLoader):
    source_type = "audio"

    def __init__(self, stt: MockSpeechToText | None = None) -> None:
        self.stt = stt or MockSpeechToText()

    def load(self, path: Path, content: bytes) -> Document:
        transcript = self.stt.transcribe(path.name, content)
        return self._document(path, transcript, {"transcript_provider": "mock"})
