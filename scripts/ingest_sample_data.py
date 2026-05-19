from __future__ import annotations

from pathlib import Path

from app.rag.service import RAGService


def main() -> None:
    service = RAGService()
    sample_dir = Path("data/sample")
    for path in sorted(sample_dir.glob("*.md")):
        result = service.ingest_file(path.name, path.read_bytes())
        print(f"ingested {path.name}: {result.chunks_added} chunks")


if __name__ == "__main__":
    main()
