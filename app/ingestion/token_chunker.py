from __future__ import annotations

import re
from dataclasses import asdict
from datetime import UTC, datetime

from app.ingestion.chunker import TextChunker
from app.ingestion.identity import identity_hash
from app.ingestion.normalizer import split_text_blocks
from app.retrieval.vector_store import chunk_search_text
from app.schemas.document import Chunk, ChunkMetadata, Document
from app.schemas.extraction import TextBlock


class TokenChunker:
    version = "token-structure-v2"

    def __init__(self, counter, target_tokens=500, overlap_tokens=80, min_tokens=100, max_tokens=640):
        if not 0 <= overlap_tokens < target_tokens <= max_tokens or min_tokens < 1:
            raise ValueError("Invalid token chunk window")
        self.counter, self.target, self.overlap, self.minimum, self.maximum = (
            counter,
            target_tokens,
            overlap_tokens,
            min_tokens,
            max_tokens,
        )

    def _sections(self, document):
        pages = document.metadata.get("extraction", {}).get("pages", [])
        if pages:
            for page in pages:
                if page["status"] == "failed":
                    continue
                title, blocks = None, []
                for data in page["blocks"]:
                    block = TextBlock(**data)
                    if block.kind == "heading":
                        if blocks:
                            yield title, page["page_number"], blocks
                        title, blocks = block.text, []
                    else:
                        blocks.append(block)
                if blocks:
                    yield title, page["page_number"], blocks
        else:
            for title, page, text in TextChunker()._split_sections(document.text):
                yield title, page, split_text_blocks(text, page or 1)

    def chunk(self, document: Document) -> list[Chunk]:
        output = []
        parents = {}
        for section_index, (title, page, blocks) in enumerate(self._sections(document)):
            parent_id = "parent-" + identity_hash([document.doc_id, str(page), str(section_index), title or ""])[:24]
            text = "\n\n".join(b.text for b in blocks).strip()
            if not text:
                continue
            block_ranges, cursor = [], 0
            for block in blocks:
                block_ranges.append((cursor, cursor + len(block.text), block))
                cursor += len(block.text) + 2
            parents[parent_id] = {
                "text": text,
                "page_number": page,
                "section_title": title,
                "blocks": [asdict(b) for b in blocks],
            }
            prefix = "\n".join(v for v in (document.title, title) if v)
            if self.counter.count(prefix) >= self.maximum // 2:
                raise ValueError("Contextual title/heading consumes too much of the chunk token budget")
            budget = self.maximum - self.counter.count(prefix) - 4
            if budget < 1:
                raise ValueError("No body token budget")
            offsets = self.counter.offsets(text)
            if not offsets:
                continue
            sentence_ends = {m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)\s+|\n\n", text)}
            block_ends = {end for _, end, _ in block_ranges}
            start_token = 0
            while start_token < len(offsets):
                stop_token = min(len(offsets), start_token + min(self.target, budget))
                start, end = offsets[start_token][0], offsets[stop_token - 1][1]
                if stop_token < len(offsets):
                    candidates = [p for p in sentence_ends | block_ends if start + 1 < p <= end]
                    if candidates:
                        end = max(candidates)
                    # Keep short tables/lists intact if the tentative cut enters one.
                    for left, right, block in block_ranges:
                        if left < end < right and block.kind in {"table", "list"}:
                            if self.counter.count(text[start:right]) <= budget:
                                end = right
                            elif left > start:
                                end = left
                    stop_token = max(
                        start_token + 1,
                        next((i for i in range(start_token, len(offsets)) if offsets[i][0] >= end), len(offsets)),
                    )
                part = text[start:end]
                trim = len(part) - len(part.lstrip())
                start += trim
                end = start + len(part.strip())
                part = text[start:end]
                spans = [
                    {
                        "page_number": block.page_number,
                        "block_id": block.block_id,
                        "start": max(start, left) - left,
                        "end": min(end, right) - left,
                    }
                    for left, right, block in block_ranges
                    if left < end and right > start
                ]
                methods = {block.method for left, right, block in block_ranges if left < end and right > start}
                confidence = [
                    block.confidence
                    for left, right, block in block_ranges
                    if left < end and right > start and block.confidence is not None
                ]
                count = self.counter.count(part)
                fingerprint = identity_hash(
                    [self.version, str(self.target), str(self.overlap), self.counter.name, self.counter.revision]
                )
                digest = identity_hash(
                    [
                        document.doc_id,
                        document.metadata.get("revision_id", ""),
                        parent_id,
                        str(start),
                        part,
                        fingerprint,
                    ]
                )
                metadata = ChunkMetadata(
                    document.doc_id,
                    document.source_path,
                    document.source_type,
                    document.title,
                    document.metadata.get("created_at") or datetime.now(UTC).isoformat(),
                    digest,
                    page_number=page,
                    section_title=title,
                    parent_id=parent_id,
                    extra={
                        "chunk_index": len(output),
                        "char_start": start,
                        "char_end": end,
                        "token_start": start_token,
                        "token_end": stop_token,
                        "token_count": count,
                        "source_spans": spans,
                        "offset_space": "normalized_section",
                        "chunker_version": fingerprint,
                        "extraction_method": next(iter(methods)) if len(methods) == 1 else "mixed",
                        "ocr_confidence": min(confidence) if confidence else None,
                        "document_hash": document.metadata.get("content_hash"),
                        "revision_id": document.metadata.get("revision_id"),
                        "warnings": ["short_semantic_unit"] if count < self.minimum else [],
                    },
                )
                chunk = Chunk("chunk-" + digest[:32], part, metadata)
                indexed_count = self.counter.count(chunk_search_text(chunk))
                if indexed_count > self.maximum:
                    raise ValueError("Chunk exceeds embedding token budget")
                metadata.extra["indexed_token_count"] = indexed_count
                if part:
                    output.append(chunk)
                if stop_token >= len(offsets):
                    break
                next_token = max(start_token + 1, stop_token - self.overlap)
                # Align overlap to a sentence boundary where available.
                overlap_positions = [i for i in range(next_token, stop_token) if offsets[i][0] in sentence_ends]
                start_token = overlap_positions[0] if overlap_positions else next_token
        document.metadata["parents"] = parents
        return output
