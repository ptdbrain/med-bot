from typing import Protocol

from app.schemas.extraction import OCRResult


class OCRProvider(Protocol):
    name: str

    def recognize(self, image: bytes, *, page_number: int, languages: tuple[str, ...]) -> OCRResult: ...
