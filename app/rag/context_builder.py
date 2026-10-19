from __future__ import annotations

import json
from dataclasses import dataclass, replace

from app.ingestion.tokenizer import ConservativeTokenCounter
from app.schemas.document import SearchResult


def context_payload(results: list[SearchResult]) -> list[dict]:
    return [
        {
            "source_id": str(i),
            "citation_label": f"[{i}]",
            "title": r.chunk.metadata.title,
            "page_number": r.chunk.metadata.page_number,
            "section_title": r.chunk.metadata.section_title,
            "text": r.chunk.text,
        }
        for i, r in enumerate(results, 1)
    ]


@dataclass
class PackedContext:
    items: list[SearchResult]
    token_count: int
    source_map: dict[str, dict]
    omitted_ids: list[str]


class ContextBuilder:
    def __init__(self, counter=None):
        self.counter = counter or ConservativeTokenCounter()

    def pack(self, query: str, results: list[SearchResult], *, input_budget: int) -> PackedContext:
        selected, omitted, seen = [], [], set()
        for item in results:
            key = " ".join(item.chunk.text.split())
            if key in seen or not key:
                omitted.append(item.chunk.chunk_id)
                continue
            candidate = selected + [item]
            serialized = json.dumps(context_payload(candidate), ensure_ascii=False)
            if self.counter.count(query + serialized) > input_budget and item.chunk.metadata.extra.get("child_excerpt"):
                excerpt = item.chunk.metadata.extra["child_excerpt"]
                start = item.chunk.text.find(excerpt)
                if start >= 0:
                    metadata = replace(
                        item.chunk.metadata,
                        extra={
                            **item.chunk.metadata.extra,
                            "context_window_char_start": start,
                            "context_window_char_end": start + len(excerpt),
                        },
                    )
                    item = replace(item, chunk=replace(item.chunk, text=excerpt, metadata=metadata))
                    candidate = selected + [item]
                    serialized = json.dumps(context_payload(candidate), ensure_ascii=False)
            if input_budget <= 0 or self.counter.count(query + serialized) > input_budget:
                omitted.append(item.chunk.chunk_id)
                continue
            selected.append(item)
            seen.add(key)
        source_map = {
            str(i): {
                "chunk_id": r.chunk.chunk_id,
                "doc_id": r.chunk.metadata.doc_id,
                "page_number": r.chunk.metadata.page_number,
                "revision_id": r.chunk.metadata.extra.get("revision_id"),
                "quote": r.chunk.text,
            }
            for i, r in enumerate(selected, 1)
        }
        total = self.counter.count(query + json.dumps(context_payload(selected), ensure_ascii=False)) if selected else 0
        return PackedContext(selected, total, source_map, omitted)
