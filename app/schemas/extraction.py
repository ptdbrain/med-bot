from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SourceSpan:
    page_number: int
    block_id: str
    start: int
    end: int

    def __post_init__(self):
        if self.page_number < 1 or self.start < 0 or self.end < self.start:
            raise ValueError("Invalid source span")


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    page_number: int
    kind: Literal["heading", "paragraph", "list", "table", "caption"]
    text: str
    bbox: tuple[float, float, float, float] | None = None
    method: Literal["text", "ocr", "mixed"] = "text"
    confidence: float | None = None

    def __post_init__(self):
        if self.page_number < 1 or self.kind not in {"heading", "paragraph", "list", "table", "caption"}:
            raise ValueError("Invalid block page or kind")
        if self.confidence is not None and (not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1):
            raise ValueError("Confidence must be between zero and one")
        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox
            if not all(math.isfinite(v) for v in self.bbox) or min(x0, y0) < 0 or x1 < x0 or y1 < y0:
                raise ValueError("Invalid bounding box")


@dataclass(frozen=True)
class OCRResult:
    blocks: tuple[TextBlock, ...]
    confidence: float | None
    warnings: tuple[str, ...] = ()


@dataclass
class PageExtraction:
    page_number: int
    classification: str
    blocks: list[TextBlock] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    status: str = "usable"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionReport:
    pages: list[PageExtraction]
    fingerprint: str = "extract-v2"

    def to_dict(self) -> dict:
        return {
            "pages": [p.to_dict() for p in self.pages],
            "fingerprint": self.fingerprint,
            "page_count": len(self.pages),
            "usable_pages": sum(p.status == "usable" for p in self.pages),
            "failed_pages": sum(p.status == "failed" for p in self.pages),
            "warning_pages": sum(bool(p.warnings) for p in self.pages),
        }
