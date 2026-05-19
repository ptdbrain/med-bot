from __future__ import annotations

from pathlib import Path

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class MockImageCaptioner:
    def caption(self, filename: str) -> str:
        return f"Mo ta anh hoc tap y khoa tu file {filename}. Can thay provider vision that neu dung production."


class ImageProcessor(DocumentLoader):
    source_type = "image"

    def __init__(self, captioner: MockImageCaptioner | None = None) -> None:
        self.captioner = captioner or MockImageCaptioner()

    def load(self, path: Path, content: bytes) -> Document:
        ocr_text = self._mock_ocr(path)
        caption = self.captioner.caption(path.name)
        text = f"OCR text:\n{ocr_text}\n\nImage caption:\n{caption}"
        return self._document(path, text, {"image_path": path.as_posix(), "caption": caption})

    def _mock_ocr(self, path: Path) -> str:
        return f"Mock OCR cho {path.name}. Hay cau hinh pytesseract hoac easyocr de trich xuat text that."
