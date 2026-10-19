import base64

from app.providers.http_transport import HTTPTransport
from app.schemas.extraction import OCRResult, TextBlock


class HTTPOCRProvider:
    name = "http"

    def __init__(self, settings, transport=None):
        self.transport = transport or HTTPTransport(settings)

    def recognize(self, image, *, page_number, languages, correlation_id=None, job_id=None, dpi=None):
        response = self.transport.post(
            "/v1/ocr",
            {
                "image_base64": base64.b64encode(image).decode(),
                "page_number": page_number,
                "languages": list(languages),
                "correlation_id": correlation_id,
                "job_id": job_id,
                "dpi": dpi,
            },
        )
        blocks = tuple(TextBlock(**b) for b in response["blocks"])
        if any(b.page_number != page_number for b in blocks):
            raise ValueError("OCR page provenance mismatch")
        return OCRResult(
            blocks,
            response.get("confidence"),
            tuple(response.get("warnings", [])),
            response.get("diagnostics", {}),
        )
