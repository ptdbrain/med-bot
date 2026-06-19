from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.ingestion.audio_processor import SpeechToTextProvider, TranscriptionResult
from app.ingestion.image_processor import ImageProcessor
from app.ingestion.video_processor import VideoProcessor
from app.main import create_app
from app.rag.service import RAGService


class FakeOCR:
    name = "fake-ocr"

    def extract_text(self, filename: str, content: bytes) -> str:
        return "Chỉ số hemoglobin trong infographic là 13 g/dL."


class FakeCaptioner:
    name = "fake-caption"

    def caption(self, filename: str, content: bytes) -> str:
        return "Infographic học tập về xét nghiệm máu."


class FakeVideoSTT(SpeechToTextProvider):
    name = "fake-video-stt"

    def transcribe(self, filename: str, content: bytes) -> TranscriptionResult:
        return TranscriptionResult(
            text="Video bài giảng giải thích hemoglobin vận chuyển oxy.",
            provider=self.name,
            language="vi",
            duration_seconds=12.5,
        )


def _settings(root: Path) -> Settings:
    return Settings(
        data_dir=root,
        index_path=root / "indexes" / "vectors.json",
        metadata_db_path=root / "indexes" / "metadata.db",
    )


class MultimodalProcessingTests(unittest.TestCase):
    def test_image_processor_uses_injected_ocr_and_caption_providers(self) -> None:
        processor = ImageProcessor(ocr=FakeOCR(), captioner=FakeCaptioner())

        document = processor.load(Path("blood-test.png"), b"fake-png")

        self.assertIn("13 g/dL", document.text)
        self.assertIn("Infographic học tập", document.text)
        self.assertEqual(document.metadata["ocr_provider"], "fake-ocr")
        self.assertEqual(document.metadata["caption_provider"], "fake-caption")
        self.assertEqual(document.metadata["image_path"], "blood-test.png")

    def test_video_processor_builds_a_transcript_document(self) -> None:
        processor = VideoProcessor(FakeVideoSTT())

        processed = processor.process(Path("lecture.mp4"), b"fake-video")

        self.assertEqual(processed.transcription.provider, "fake-video-stt")
        self.assertEqual(processed.document.source_type, "video")
        self.assertIn("hemoglobin", processed.document.text)
        self.assertEqual(
            processed.document.metadata["video_path"],
            "lecture.mp4",
        )

    def test_video_ingest_becomes_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                _settings(root),
                stt_provider=FakeVideoSTT(),
            )

            ingest, transcription = service.ingest_video(
                "lecture.mp4",
                b"fake-video",
            )
            chat = service.chat(
                "Hemoglobin vận chuyển gì?",
                rag_mode="hybrid",
                top_k=3,
            )

        self.assertEqual(transcription.provider, "fake-video-stt")
        self.assertEqual(ingest.source_type, "video")
        self.assertTrue(chat.sources)
        self.assertEqual(chat.sources[0]["source_type"], "video")

    def test_invalid_video_extension_is_rejected_without_persisting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            service = RAGService(
                _settings(root),
                stt_provider=FakeVideoSTT(),
            )

            with self.assertRaisesRegex(ValueError, "Unsupported file extension"):
                service.ingest_video("lecture.exe", b"fake-video")

            self.assertFalse((root / "raw" / "video" / "lecture.exe").exists())

    def test_video_ingest_api_is_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(_settings(Path(tmp))))

            response = client.post(
                "/video/ingest",
                files={
                    "file": (
                        "lecture.mp4",
                        b"fake-video",
                        "video/mp4",
                    )
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["source_type"], "video")
        self.assertIn("transcription", response.json())


if __name__ == "__main__":
    unittest.main()
