from __future__ import annotations

import re
from datetime import datetime, timezone

from app.schemas.document import Chunk, ChunkMetadata, Document
from app.utils.hashing import short_hash, stable_hash


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page:(\d+)\s*-->")


class TextChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_sections(document.text)
        chunks: list[Chunk] = []
        created_at = document.metadata.get("created_at") or datetime.now(timezone.utc).isoformat()

        for section_title, page_number, text in sections:
            for part in self._window(text):
                cleaned = part.strip()
                if not cleaned:
                    continue
                chunk_hash = stable_hash(f"{document.doc_id}:{section_title}:{page_number}:{cleaned}")
                chunk_id = f"{document.doc_id}-{short_hash(chunk_hash)}"
                metadata = ChunkMetadata(
                    doc_id=document.doc_id,
                    source_path=document.source_path,
                    source_type=document.source_type,
                    title=document.title,
                    page_number=page_number,
                    section_title=section_title,
                    created_at=created_at,
                    hash=chunk_hash,
                    image_path=document.metadata.get("image_path"),
                    extra={k: v for k, v in document.metadata.items() if k not in {"created_at", "image_path"}},
                )
                chunks.append(Chunk(chunk_id=chunk_id, text=cleaned, metadata=metadata))
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str | None, int | None, str]]:
        current_heading: str | None = None
        current_page: int | None = None
        buffer: list[str] = []
        sections: list[tuple[str | None, int | None, str]] = []

        def flush() -> None:
            if buffer:
                sections.append((current_heading, current_page, "\n".join(buffer).strip()))
                buffer.clear()

        for line in text.splitlines():
            page_match = PAGE_MARKER_PATTERN.search(line)
            if page_match:
                flush()
                current_page = int(page_match.group(1))
                continue

            heading_match = HEADING_PATTERN.match(line.strip())
            if heading_match:
                flush()
                current_heading = heading_match.group(2).strip()
                buffer.append(current_heading)
            else:
                buffer.append(line)
        flush()
        if not sections and text.strip():
            sections.append((current_heading, current_page, text.strip()))
        return sections

    def _window(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            if end < len(text):
                paragraph_break = text.rfind("\n\n", start, end)
                sentence_break = text.rfind(". ", start, end)
                split_at = max(paragraph_break, sentence_break)
                if split_at > start + self.chunk_size // 2:
                    end = split_at + 1
            parts.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return parts
