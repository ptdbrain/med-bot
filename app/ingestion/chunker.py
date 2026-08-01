from __future__ import annotations

import re
from datetime import UTC, datetime

from app.schemas.document import Chunk, ChunkMetadata, Document
from app.utils.hashing import short_hash, stable_hash

HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
PAGE_MARKER_PATTERN = re.compile(r"<!--\s*page:(\d+)\s*-->")
BOUNDARY_PATTERN = re.compile(r"(?:\n\s*\n|(?<=[.!?])[\]\"')]*\s+|\n)")


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        min_chunk_size: int | None = None,
    ) -> None:
        if min_chunk_size is None:
            min_chunk_size = min(180, max(1, chunk_size // 4))
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if min_chunk_size <= 0 or min_chunk_size > chunk_size:
            raise ValueError("min_chunk_size must be between 1 and chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._split_sections(document.text)
        chunks: list[Chunk] = []
        created_at = document.metadata.get("created_at") or datetime.now(UTC).isoformat()

        for section_title, page_number, text in sections:
            parent_key = f"{document.doc_id}:{section_title or '__document__'}:" f"{page_number or 0}"
            parent_id = f"parent-{short_hash(parent_key)}"
            for chunk_index, (part, char_start, char_end) in enumerate(self._window(text)):
                cleaned = part.strip()
                if not cleaned:
                    continue
                chunk_hash = stable_hash(f"{document.doc_id}:{section_title}:{page_number}:" f"{char_start}:{cleaned}")
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
                    parent_id=parent_id,
                    extra={
                        **{k: v for k, v in document.metadata.items() if k not in {"created_at", "image_path"}},
                        "chunk_index": chunk_index,
                        "char_start": char_start,
                        "char_end": char_end,
                    },
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

    def _window(self, text: str) -> list[tuple[str, int, int]]:
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        boundaries = [match.end() for match in BOUNDARY_PATTERN.finditer(text)]
        parts: list[tuple[str, int, int]] = []
        start = 0
        while start < len(text):
            hard_end = min(len(text), start + self.chunk_size)
            lower_bound = start + min(
                self.min_chunk_size,
                max(1, self.chunk_size // 2),
            )
            candidates = [boundary for boundary in boundaries if lower_bound <= boundary <= hard_end]
            end = candidates[-1] if candidates else hard_end
            overlap_target = max(start + 1, end - self.chunk_overlap)
            aligned_starts = [boundary for boundary in boundaries if overlap_target <= boundary < end]
            next_start = aligned_starts[0] if aligned_starts else overlap_target
            if 0 < len(text) - next_start < self.min_chunk_size:
                end = len(text)

            parts.append((text[start:end], start, end))
            if end >= len(text):
                break
            if next_start <= start:
                next_start = end
            start = next_start
        return parts
