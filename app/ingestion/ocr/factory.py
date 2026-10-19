from app.config import Settings


def create_ocr_provider(settings: Settings):
    if settings.ocr_provider in {"none", "mock"}:
        return None
    if settings.ocr_provider == "paddle":
        from app.ingestion.ocr.paddle import PaddleOCRProvider

        return PaddleOCRProvider(max_pixels=settings.ocr_max_pixels)
    if settings.ocr_provider == "http":
        from app.ingestion.ocr.http import HTTPOCRProvider

        return HTTPOCRProvider(settings)
    raise ValueError("Unsupported OCR_PROVIDER")
