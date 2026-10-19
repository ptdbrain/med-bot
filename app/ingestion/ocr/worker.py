"""Process boundary avoids Paddle/PyTorch DLL conflicts on Windows."""

import base64
import contextlib
import json
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version


def _package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"


def _diagnostics(request, *, elapsed, worker_request):
    return {
        "correlation_id": request.get("correlation_id"),
        "job_id": request.get("job_id"),
        "request_id": request.get("request_id"),
        "page_number": request.get("page_number"),
        "dpi": request.get("dpi"),
        "languages": request.get("languages"),
        "image_dimensions_before_resize": request.get("image_dimensions_before_resize"),
        "image_dimensions_after_resize": request.get("image_dimensions_after_resize"),
        "model_revision": "PP-OCRv5",
        "timeout_seconds": request.get("timeout_seconds"),
        "max_pages_per_worker": request.get("max_pages_per_worker"),
        "worker_request": worker_request,
        "inference_seconds": round(elapsed, 4),
        "paddleocr_version": _package_version("paddleocr"),
        "paddlepaddle_version": _package_version("paddlepaddle"),
    }


def _error_response(request, category, exc, *, elapsed, worker_request):
    return {
        "request_id": request.get("request_id"),
        "correlation_id": request.get("correlation_id"),
        "error": {
            "category": category,
            "type": type(exc).__name__,
            "message": str(exc)[:300],
            "diagnostics": _diagnostics(request, elapsed=elapsed, worker_request=worker_request),
        },
    }


def serve(source, output, provider_factory=None, *, max_requests=64):
    provider = None
    for worker_request, line in enumerate(source, 1):
        if not line.strip():
            continue
        request = {}
        started = time.monotonic()
        try:
            decoded = json.loads(line)
            if not isinstance(decoded, dict):
                raise TypeError("request must be a JSON object")
            request = decoded
            image = base64.b64decode(request["image"], validate=True)
            if not request.get("request_id") or not request.get("correlation_id"):
                raise ValueError("request_id and correlation_id are required")
            page_number = int(request["page_number"])
            languages = tuple(request["languages"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, base64.binascii.Error) as exc:
            response = _error_response(
                request, "invalid_request", exc, elapsed=time.monotonic() - started, worker_request=worker_request
            )
        else:
            try:
                if provider is None:
                    if provider_factory is None:
                        from app.ingestion.ocr.paddle import PaddleOCRProvider

                        provider_factory = lambda: PaddleOCRProvider(isolated=False)
                    provider = provider_factory()
                with contextlib.redirect_stdout(sys.stderr):
                    result = provider.recognize(
                        image,
                        page_number=page_number,
                        languages=languages,
                        correlation_id=request["correlation_id"],
                        job_id=request.get("job_id"),
                        dpi=request.get("dpi"),
                    )
                payload = asdict(result) if hasattr(result, "__dataclass_fields__") else result
                payload["diagnostics"].update(
                    _diagnostics(request, elapsed=time.monotonic() - started, worker_request=worker_request)
                )
                response = {
                    "request_id": request["request_id"],
                    "correlation_id": request["correlation_id"],
                    "result": payload,
                }
            except Exception as exc:  # noqa: BLE001 - Return provider failures to the supervisor.
                response = _error_response(
                    request, "provider_error", exc, elapsed=time.monotonic() - started, worker_request=worker_request
                )
        response["restart_required"] = worker_request >= max_requests
        output.write(json.dumps(response, ensure_ascii=True) + "\n")
        output.flush()
        if response["restart_required"]:
            break


def main():
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
