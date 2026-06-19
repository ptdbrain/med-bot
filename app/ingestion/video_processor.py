from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.audio_processor import (
    MockSpeechToText,
    SpeechToTextProvider,
    TranscriptionResult,
)
from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


@dataclass(frozen=True)
class ProcessedVideo:
    document: Document
    transcription: TranscriptionResult


class VideoProcessor(DocumentLoader):
    source_type = "video"

    def __init__(self, stt: SpeechToTextProvider | None = None) -> None:
        self.stt = stt or MockSpeechToText()

    def process(self, path: Path, content: bytes) -> ProcessedVideo:
        if not content:
            raise ValueError("Video file is empty.")
        transcript = self.stt.transcribe(path.name, content)
        cleaned_text = " ".join(transcript.text.split()).strip()
        if not cleaned_text:
            raise ValueError("Speech-to-text provider returned an empty video transcript.")
        document = self._document(
            path,
            "Video transcript:\n" + cleaned_text,
            {
                "video_path": path.as_posix(),
                "transcript_provider": transcript.provider,
                "language": transcript.language,
                "duration_seconds": transcript.duration_seconds,
                "keyframes": [],
            },
        )
        return ProcessedVideo(document=document, transcription=transcript)

    def load(self, path: Path, content: bytes) -> Document:
        return self.process(path, content).document
