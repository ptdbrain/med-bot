import base64
import io
import json
import subprocess
import sys

from app.schemas.extraction import OCRResult, TextBlock


def limit_image_pixels(image: bytes, *, max_pixels: int) -> bytes:
    if max_pixels <= 0:
        raise ValueError("max_pixels must be positive")
    from PIL import Image

    with Image.open(io.BytesIO(image)) as source:
        width, height = source.size
        if width * height <= max_pixels:
            return image
        scale = (max_pixels / (width * height)) ** 0.5
        resized = source.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        resized.save(output, format="PNG")
        resized.close()
        return output.getvalue()


class PaddleOCRProvider:
    name = "paddle"

    def __init__(self, model=None, *, isolated=True, max_pixels=2_000_000):
        self._model = model
        self.isolated = isolated
        self.max_pixels = max_pixels

    def recognize(self, image: bytes, *, page_number: int, languages: tuple[str, ...]) -> OCRResult:
        image = limit_image_pixels(image, max_pixels=self.max_pixels)
        if self.isolated and self._model is None:
            request = json.dumps(
                {
                    "image": base64.b64encode(image).decode("ascii"),
                    "page_number": page_number,
                    "languages": languages,
                }
            )
            try:
                process = subprocess.run(
                    [sys.executable, "-m", "app.ingestion.ocr.worker"],
                    input=request,
                    text=True,
                    encoding="utf-8",
                    capture_output=True,
                    timeout=180,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("OCR worker timed out") from exc
            if process.returncode:
                raise RuntimeError("OCR worker failed; check OCR installation and model availability")
            result = json.loads(process.stdout)
            if "error" in result:
                raise RuntimeError(
                    f"OCR worker failed ({result['error']}); check OCR installation and model availability"
                )
            return OCRResult(
                tuple(TextBlock(**b) for b in result["blocks"]), result["confidence"], tuple(result["warnings"])
            )
        import numpy as np
        from PIL import Image

        if self._model is None:
            from paddleocr import PaddleOCR

            self._model = PaddleOCR(
                lang="vi" if "vi" in languages else "en",
                ocr_version="PP-OCRv5",
                use_doc_orientation_classify=True,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                device="cpu",
                enable_mkldnn=False,
                cpu_threads=4,
            )
        with Image.open(io.BytesIO(image)) as im:
            pixels = np.array(im.convert("RGB"))
        height, width = pixels.shape[:2]
        blocks = []
        for output in self._model.predict(pixels):
            payload = output.json if hasattr(output, "json") else output
            if callable(payload):
                payload = payload()
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = payload.get("res", payload)
            for text, score, polygon in zip(
                payload["rec_texts"], payload["rec_scores"], payload["rec_polys"], strict=True
            ):
                if not text.strip():
                    continue
                xs, ys = [float(p[0]) / width for p in polygon], [float(p[1]) / height for p in polygon]
                blocks.append(
                    TextBlock(
                        f"p{page_number}-ocr{len(blocks)}",
                        page_number,
                        "paragraph",
                        text,
                        (max(0, min(xs)), max(0, min(ys)), max(xs), max(ys)),
                        "ocr",
                        float(score),
                    )
                )
        confidence = (
            sum(len(b.text) * b.confidence for b in blocks) / sum(len(b.text) for b in blocks) if blocks else None
        )
        warnings = ("low_ocr_confidence",) if confidence is not None and confidence < 0.7 else ()
        return OCRResult(tuple(blocks), confidence, warnings)
