from __future__ import annotations

import io
import tempfile
import unittest
import wave
from pathlib import Path

from app.config import Settings
from app.ingestion.audio_processor import (
    AudioProcessor,
    MockSpeechToText,
    SpeechToTextProvider,
    TranscriptionResult,
    create_speech_to_text,
)
from app.rag.service import RAGService


def _wav_bytes(duration_seconds: float = 0.1, sample_rate: int = 8000) -> bytes:
    frame_count = int(duration_seconds * sample_rate)
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


class FakeSpeechToText(SpeechToTextProvider):
    name = "fake"

    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        return TranscriptionResult(
            text=self.transcript,
            provider=self.name,
            language="vi",
            duration_seconds=None,
            segments=[],
        )


class AudioProcessingTest(unittest.TestCase):
    def test_factory_builds_configured_mock_provider(self) -> None:
        provider = create_speech_to_text(
            Settings(
                stt_provider="mock",
                mock_stt_transcript="Transcript cấu hình.",
                stt_language="vi",
            )
        )

        result = provider.transcribe("sample.wav", _wav_bytes())

        self.assertIsInstance(provider, MockSpeechToText)
        self.assertEqual(result.text, "Transcript cấu hình.")
        self.assertEqual(result.language, "vi")

    def test_processor_extracts_wav_metadata_and_builds_document(self) -> None:
        processor = AudioProcessor(FakeSpeechToText("Bài giảng nói về hemoglobin."))

        processed = processor.process(Path("lecture.wav"), _wav_bytes())

        self.assertEqual(processed.transcription.provider, "fake")
        self.assertEqual(processed.audio_info.sample_rate, 8000)
        self.assertEqual(processed.audio_info.channels, 1)
        self.assertAlmostEqual(processed.audio_info.duration_seconds or 0, 0.1, places=2)
        self.assertIn("hemoglobin", processed.document.text)
        self.assertEqual(processed.document.metadata["transcript_provider"], "fake")

    def test_processor_rejects_invalid_wav_before_transcription(self) -> None:
        processor = AudioProcessor(FakeSpeechToText("unused"))

        with self.assertRaisesRegex(ValueError, "Invalid WAV"):
            processor.process(Path("broken.wav"), b"not-a-wave-file")

    def test_voice_chat_transcribes_question_then_runs_rag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vector_store.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                ),
                stt_provider=FakeSpeechToText("FAST trong đột quỵ gồm dấu hiệu nào?"),
            )
            service.ingest_file(
                "stroke.md",
                "# FAST\n\nFAST gồm méo miệng, yếu tay, nói khó và cần gọi cấp cứu sớm.".encode(),
            )

            transcription, chat = service.voice_chat(
                "question.wav",
                _wav_bytes(),
                rag_mode="hybrid",
                top_k=3,
            )

        self.assertEqual(transcription.provider, "fake")
        self.assertEqual(chat.mode_used, "hybrid")
        self.assertTrue(chat.sources)
        self.assertIn("stroke.md", chat.sources[0]["source_path"])

    def test_audio_lecture_becomes_searchable_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vector_store.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                ),
                stt_provider=FakeSpeechToText(
                    "Hemoglobin phản ánh khả năng vận chuyển oxy của hồng cầu trong xét nghiệm máu."
                ),
            )

            ingest_result, transcription = service.ingest_audio("lecture.wav", _wav_bytes())
            chat = service.chat("Hemoglobin có vai trò gì?", rag_mode="hybrid", top_k=3)

        self.assertEqual(transcription.provider, "fake")
        self.assertGreater(ingest_result.chunks_added, 0)
        self.assertTrue(chat.sources)
        self.assertEqual(chat.sources[0]["source_type"], "audio")
        self.assertIn("lecture.wav", chat.sources[0]["source_path"])

    def test_invalid_audio_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                Settings(
                    data_dir=root,
                    index_path=root / "indexes" / "vector_store.json",
                    metadata_db_path=root / "indexes" / "metadata.db",
                ),
                stt_provider=FakeSpeechToText("unused"),
            )

            with self.assertRaisesRegex(ValueError, "Invalid WAV"):
                service.ingest_audio("broken.wav", b"not-a-wave-file")

            saved_path = root / "raw" / "audio" / "broken.wav"
            self.assertFalse(saved_path.exists())


if __name__ == "__main__":
    unittest.main()
