"""Process boundary avoids Paddle/PyTorch DLL conflicts on Windows."""

import base64
import contextlib
import json
import sys
from dataclasses import asdict


def serve(source, output, provider_factory=None):
    provider = None
    for line in source:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if provider is None:
                if provider_factory is None:
                    from app.ingestion.ocr.paddle import PaddleOCRProvider

                    provider_factory = lambda: PaddleOCRProvider(isolated=False)
                provider = provider_factory()
            with contextlib.redirect_stdout(sys.stderr):
                result = provider.recognize(
                    base64.b64decode(request["image"], validate=True),
                    page_number=int(request["page_number"]),
                    languages=tuple(request["languages"]),
                )
            payload = asdict(result) if hasattr(result, "__dataclass_fields__") else result
            output.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception as exc:  # noqa: BLE001 - Isolate page errors and keep the worker available.
            output.write(json.dumps({"error": type(exc).__name__}) + "\n")
        output.flush()


def main():
    serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    main()
