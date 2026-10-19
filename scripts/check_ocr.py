import argparse
import json
import time
from pathlib import Path

from app.ingestion.ocr.paddle import PaddleOCRProvider
from app.ingestion.pdf_loader import PdfLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("data/reports/ocr-live.json"))
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()
    report = []
    for path in args.files:
        started = time.perf_counter()
        try:
            doc = PdfLoader(ocr=PaddleOCRProvider(), dpi=args.dpi).load(path, path.read_bytes())
            report.append(
                {
                    "file": str(path),
                    "text": doc.text,
                    "extraction": doc.metadata["extraction"],
                    "seconds": time.perf_counter() - started,
                    "status": "ok",
                }
            )
        except (RuntimeError, ValueError) as exc:
            report.append(
                {"file": str(path), "status": "failed", "error": str(exc), "seconds": time.perf_counter() - started}
            )
        print(json.dumps({k: v for k, v in report[-1].items() if k not in {"text", "extraction"}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return int(any(row["status"] != "ok" for row in report))


if __name__ == "__main__":
    raise SystemExit(main())
