from __future__ import annotations

import io
import os
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from app.config import Settings
from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


@dataclass(frozen=True)
class TranscriptionSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    provider: str
    language: str | None = None
    duration_seconds: float | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "provider": self.provider,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass(frozen=True)
class AudioInfo:
    format: str
    duration_seconds: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    sample_width: int | None = None
    size_bytes: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProcessedAudio:
    document: Document
    transcription: TranscriptionResult
    audio_info: AudioInfo


class SpeechToTextProvider(ABC):
    name: str

    @abstractmethod
    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        raise NotImplementedError


class MockSpeechToText(SpeechToTextProvider):
    name = "mock"

    def __init__(
        self,
        transcript: str = "Đây là transcript mẫu. Hãy cấu hình faster-whisper để nhận dạng audio thật.",
        language: str = "vi",
    ) -> None:
        self.transcript = transcript
        self.language = language

    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.transcript,
            provider=self.name,
            language=self.language,
            duration_seconds=None,
            segments=[],
        )


class FasterWhisperSpeechToText(SpeechToTextProvider):
    name = "faster-whisper"

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = "vi",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        model = self._get_model()
        suffix = Path(filename).suffix.lower() or ".wav"
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as audio_file:
                temporary_path = audio_file.name
                audio_file.write(content)
            segments_iterator, info = model.transcribe(
                temporary_path,
                language=self.language,
                vad_filter=True,
            )
            segments = [
                TranscriptionSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                )
                for segment in segments_iterator
                if segment.text.strip()
            ]
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass

        text = " ".join(segment.text for segment in segments).strip()
        duration = float(getattr(info, "duration", 0.0) or 0.0) or None
        language = getattr(info, "language", None) or self.language
        return TranscriptionResult(
            text=text,
            provider=self.name,
            language=language,
            duration_seconds=duration,
            segments=segments,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "STT_PROVIDER=faster-whisper requires the optional 'faster-whisper' dependency."
            ) from exc
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model


def create_speech_to_text(settings: Settings) -> SpeechToTextProvider:
    provider = settings.stt_provider.strip().lower()
    if provider == "mock":
        return MockSpeechToText(
            transcript=settings.mock_stt_transcript,
            language=settings.stt_language or "vi",
        )
    if provider in {"faster-whisper", "faster_whisper"}:
        return FasterWhisperSpeechToText(
            model_size=settings.whisper_model_size,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.stt_language,
        )
    raise ValueError(f"Unsupported STT provider: {settings.stt_provider}")


class AudioProcessor(DocumentLoader):
    source_type = "audio"

    def __init__(self, stt: SpeechToTextProvider | None = None) -> None:
        self.stt = stt or MockSpeechToText()

    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        transcription, _ = self._transcribe_with_info(filename, content)
        return transcription

    def process(self, path: Path, content: bytes) -> ProcessedAudio:
        transcription, audio_info = self._transcribe_with_info(path.name, content)
        metadata = {
            "audio_path": path.as_posix(),
            "transcript_provider": transcription.provider,
            "language": transcription.language,
            "duration_seconds": transcription.duration_seconds,
            "segments": [segment.to_dict() for segment in transcription.segments],
            "audio_info": audio_info.to_dict(),
        }
        document = self._document(path, transcription.text, metadata)
        return ProcessedAudio(
            document=document,
            transcription=transcription,
            audio_info=audio_info,
        )

    def load(self, path: Path, content: bytes) -> Document:
        return self.process(path, content).document

    def _transcribe_with_info(
        self,
        filename: str,
        content: bytes,
    ) -> tuple[TranscriptionResult, AudioInfo]:
        audio_info = inspect_audio(filename, content)
        transcription = self.stt.transcribe(filename, content)
        cleaned_text = " ".join(transcription.text.split()).strip()
        if not cleaned_text:
            raise ValueError("Speech-to-text provider returned an empty transcript.")
        duration = transcription.duration_seconds or audio_info.duration_seconds
        return replace(transcription, text=cleaned_text, duration_seconds=duration), audio_info


def inspect_audio(filename: str, content: bytes) -> AudioInfo:
    if not content:
        raise ValueError("Audio file is empty.")

    extension = Path(filename).suffix.lower()
    if extension != ".wav":
        return AudioInfo(format=extension.lstrip(".") or "unknown", size_bytes=len(content))

    try:
        with wave.open(io.BytesIO(content), "rb") as audio:
            frame_count = audio.getnframes()
            sample_rate = audio.getframerate()
            duration = frame_count / sample_rate if sample_rate else 0.0
            return AudioInfo(
                format="wav",
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=audio.getnchannels(),
                sample_width=audio.getsampwidth(),
                size_bytes=len(content),
            )
    except (EOFError, wave.Error) as exc:
        raise ValueError("Invalid WAV file.") from exc
