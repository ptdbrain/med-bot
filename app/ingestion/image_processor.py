from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.ingestion.base import DocumentLoader
from app.schemas.document import Document


class OCRProvider(Protocol):
    name: str

    def extract_text(self, filename: str, content: bytes) -> str: ...


class ImageCaptionProvider(Protocol):
    name: str

    def caption(self, filename: str, content: bytes) -> str: ...


class MockOCR:
    name = "mock"

    def extract_text(self, filename: str, content: bytes) -> str:
        return f"Mock OCR cho {filename}. " "Hãy cấu hình pytesseract hoặc easyocr để trích xuất text thật."


class MockImageCaptioner:
    name = "mock"

    def caption(self, filename: str, content: bytes) -> str:
        return f"Mo ta anh hoc tap y khoa tu file {filename}. Can thay provider vision that neu dung production."


class ImageProcessor(DocumentLoader):
    source_type = "image"

    def __init__(
        self,
        ocr: OCRProvider | None = None,
        captioner: ImageCaptionProvider | None = None,
    ) -> None:
        self.ocr = ocr or MockOCR()
        self.captioner = captioner or MockImageCaptioner()

    def load(self, path: Path, content: bytes) -> Document:
        if hasattr(self.ocr, "recognize"):
            from dataclasses import asdict

            result = self.ocr.recognize(content, page_number=1, languages=("vi", "en"))
            ocr_text = "\n".join(block.text for block in result.blocks).strip()
            if not ocr_text:
                raise ValueError("OCR returned no usable text")
            return self._document(
                path,
                ocr_text,
                {
                    "image_path": path.as_posix(),
                    "ocr_provider": self.ocr.name,
                    "ocr_confidence": result.confidence,
                    "extraction": {
                        "pages": [
                            {
                                "page_number": 1,
                                "classification": "scan",
                                "status": "usable",
                                "blocks": [asdict(block) for block in result.blocks],
                                "warnings": list(result.warnings),
                            }
                        ]
                    },
                },
            )
        ocr_text = self.ocr.extract_text(path.name, content).strip()
        caption = self.captioner.caption(path.name, content).strip()
        if not ocr_text and not caption:
            raise ValueError("OCR and caption providers returned empty content.")
        text = f"OCR text:\n{ocr_text}\n\nImage caption:\n{caption}"
        return self._document(
            path,
            text,
            {
                "image_path": path.as_posix(),
                "caption": caption,
                "ocr_provider": self.ocr.name,
                "caption_provider": self.captioner.name,
            },
        )
