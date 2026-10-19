from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

from app.schemas.extraction import OCRResult, TextBlock

logger = logging.getLogger(__name__)
_WINDOWS_NATIVE_CRASH_CODES = {0xC0000005, 0xC0000409, 0xC000001D, 0xC00000FD}


class OCRWorkerError(RuntimeError):
    def __init__(self, message: str, diagnostics: dict, *, retryable: bool = False):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.retryable = retryable


def image_dimensions(image: bytes) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(image)) as source:
        return source.size


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

    def __init__(self, model=None, *, isolated=True, max_pixels=2_000_000, timeout=180, max_pages_per_worker=64):
        if max_pages_per_worker < 1 or timeout <= 0:
            raise ValueError("OCR worker limits must be positive")
        self._model = model
        self.isolated = isolated
        self.max_pixels = max_pixels
        self.timeout = timeout
        self.max_pages_per_worker = max_pages_per_worker
        self._process = None
        self._responses = None
        self._stderr_tail = deque(maxlen=32)
        self._stdout_reader = None
        self._stderr_reader = None
        self._lock = threading.RLock()
        self._worker_pages = 0

    def _start_worker(self):
        stderr_tail = deque(maxlen=32)
        self._stderr_tail = stderr_tail
        process = subprocess.Popen(
            [sys.executable, "-m", "app.ingestion.ocr.worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        responses = queue.Queue()
        self._process, self._responses, self._stderr_tail = process, responses, stderr_tail
        self._worker_pages = 0

        def read_stdout():
            for line in process.stdout:
                responses.put(line)
            responses.put(None)

        def read_stderr():
            for line in process.stderr:
                stderr_tail.append(line.rstrip()[-500:])

        self._stdout_reader = threading.Thread(target=read_stdout, daemon=True)
        self._stdout_reader.start()
        self._stderr_reader = threading.Thread(target=read_stderr, daemon=True)
        self._stderr_reader.start()

    def _stop_worker(self, *, force=False):
        process = self._process
        self._process = self._responses = None
        if process is None:
            return
        try:
            if force:
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
            elif process.stdin:
                process.stdin.close()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    process.terminate()
                except OSError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                    except OSError:
                        pass
                    process.wait(timeout=2)
        finally:
            if self._stdout_reader is not None:
                self._stdout_reader.join(timeout=0.25)
                self._stdout_reader = None
            if self._stderr_reader is not None:
                self._stderr_reader.join(timeout=0.25)
                self._stderr_reader = None
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def close_document(self):
        with self._lock:
            self._stop_worker()

    @contextlib.contextmanager
    def document_session(self):
        self._lock.acquire()
        try:
            yield
        finally:
            try:
                self._stop_worker()
            finally:
                self._lock.release()

    def _diagnostics(self, request, *, category=None, returncode=None, attempt=0, retryable=False):
        details = {
            "correlation_id": request.get("correlation_id"),
            "job_id": request.get("job_id"),
            "request_id": request.get("request_id"),
            "page_number": request.get("page_number"),
            "dpi": request.get("dpi"),
            "image_dimensions_before_resize": request.get("image_dimensions_before_resize"),
            "image_dimensions_after_resize": request.get("image_dimensions_after_resize"),
            "model_revision": request.get("model_revision"),
            "attempt": attempt,
            "retryable": retryable,
            "timeout_seconds": request.get("timeout_seconds"),
            "max_pages_per_worker": request.get("max_pages_per_worker"),
            "returncode": returncode,
            "stderr_tail": "\n".join(self._stderr_tail),
        }
        if category:
            details["failure_category"] = category
        return details

    def _fail(self, message, request, category, *, retryable=False, attempt=0):
        returncode = self._process.poll() if self._process is not None else None
        if returncode is not None:
            unsigned = returncode & 0xFFFFFFFF
            if unsigned in _WINDOWS_NATIVE_CRASH_CODES or returncode in {-11, -6}:
                category = "native_worker_exit"
        diagnostics = self._diagnostics(
            request, category=category, returncode=returncode, attempt=attempt, retryable=retryable
        )
        diagnostics["error_message"] = message[:500]
        logger.error("ocr_worker_failure %s", json.dumps(diagnostics, ensure_ascii=True, sort_keys=True))
        return OCRWorkerError(f"OCR {category} on page {request.get('page_number')}", diagnostics, retryable=retryable)

    def _recognize_isolated(self, request):
        if self._process is not None and self._process.poll() is not None:
            self._stop_worker()
        if self._process is None:
            try:
                self._start_worker()
            except OSError as exc:
                raise self._fail(str(exc), request, "worker_start_failed") from exc
        try:
            self._process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
            self._process.stdin.flush()
            response_line = self._responses.get(timeout=self.timeout)
        except queue.Empty:
            raise self._fail(
                "OCR worker timed out", request, "timeout", retryable=True, attempt=request.get("attempt", 0)
            )
        except (BrokenPipeError, OSError) as exc:
            raise self._fail(str(exc), request, "worker_pipe_error") from exc
        if response_line is None:
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            if self._stderr_reader is not None:
                self._stderr_reader.join(timeout=0.25)
            raise self._fail(
                "OCR worker exited without a response",
                request,
                "worker_exit",
                attempt=request.get("attempt", 0),
            )
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise self._fail(
                "OCR worker returned malformed JSON",
                request,
                "worker_protocol_error",
                attempt=request.get("attempt", 0),
            ) from exc
        if not isinstance(response, dict):
            raise self._fail(
                "OCR worker returned a non-object response",
                request,
                "worker_protocol_error",
                attempt=request.get("attempt", 0),
            )
        if response.get("request_id") != request["request_id"]:
            raise self._fail(
                "OCR worker response ID mismatch", request, "worker_protocol_error", attempt=request.get("attempt", 0)
            )
        if response.get("error"):
            failure = response["error"]
            diagnostics = {**failure.get("diagnostics", {}), **self._diagnostics(request)}
            diagnostics["failure_category"] = failure.get("category", "provider_error")
            diagnostics["error_type"] = failure.get("type")
            diagnostics["error_message"] = failure.get("message")
            logger.error("ocr_worker_failure %s", json.dumps(diagnostics, ensure_ascii=True, sort_keys=True))
            raise OCRWorkerError(
                f"OCR {diagnostics['failure_category']} on page {request.get('page_number')}", diagnostics
            )
        payload = response.get("result")
        if not isinstance(payload, dict) or not isinstance(payload.get("blocks"), list):
            raise self._fail(
                "OCR worker returned invalid result schema",
                request,
                "worker_protocol_error",
                attempt=request.get("attempt", 0),
            )
        diagnostics = {**self._diagnostics(request), **payload.get("diagnostics", {})}
        diagnostics["worker_request"] = self._worker_pages + 1
        self._worker_pages += 1
        logger.info("ocr_page_complete %s", json.dumps(diagnostics, ensure_ascii=True, sort_keys=True))
        result = OCRResult(
            tuple(TextBlock(**block) for block in payload["blocks"]),
            payload.get("confidence"),
            tuple(payload.get("warnings", [])),
            diagnostics,
        )
        if response.get("restart_required") or self._worker_pages >= self.max_pages_per_worker:
            self._stop_worker()
        return result

    def recognize(
        self,
        image: bytes,
        *,
        page_number: int,
        languages: tuple[str, ...],
        correlation_id: str | None = None,
        job_id: str | None = None,
        dpi: int | None = None,
    ) -> OCRResult:
        dimensions_before = image_dimensions(image)
        bounded = limit_image_pixels(image, max_pixels=self.max_pixels)
        dimensions_after = image_dimensions(bounded)
        correlation_id = correlation_id or str(uuid.uuid4())
        if self.isolated and self._model is None:
            request = {
                "request_id": str(uuid.uuid4()),
                "correlation_id": correlation_id,
                "job_id": job_id,
                "page_number": page_number,
                "languages": languages,
                "dpi": dpi,
                "image_dimensions_before_resize": dimensions_before,
                "image_dimensions_after_resize": dimensions_after,
                "model_revision": "PP-OCRv5",
                "timeout_seconds": self.timeout,
                "max_pages_per_worker": self.max_pages_per_worker,
                "image": base64.b64encode(bounded).decode("ascii"),
            }
            with self._lock:
                for attempt in range(2):
                    request["attempt"] = attempt
                    try:
                        result = self._recognize_isolated(request)
                        result.diagnostics["attempt"] = attempt
                        return result
                    except OCRWorkerError as exc:
                        if not exc.retryable or attempt == 1:
                            self._stop_worker(force=True)
                            raise
                        self._stop_worker(force=True)
            raise AssertionError("unreachable")
        started = time.monotonic()
        result = self._recognize_in_process(bounded, page_number=page_number, languages=languages)
        result.diagnostics.update(
            {
                "correlation_id": correlation_id,
                "job_id": job_id,
                "page_number": page_number,
                "dpi": dpi,
                "image_dimensions_before_resize": dimensions_before,
                "image_dimensions_after_resize": dimensions_after,
                "model_revision": "PP-OCRv5",
                "inference_seconds": round(time.monotonic() - started, 4),
            }
        )
        return result

    def _recognize_in_process(self, image: bytes, *, page_number: int, languages: tuple[str, ...]) -> OCRResult:
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
            sum(len(block.text) * block.confidence for block in blocks) / sum(len(block.text) for block in blocks)
            if blocks
            else None
        )
        warnings = ("low_ocr_confidence",) if confidence is not None and confidence < 0.7 else ()
        return OCRResult(tuple(blocks), confidence, warnings)
