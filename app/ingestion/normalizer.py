import re
import unicodedata
from collections import Counter
from dataclasses import replace

from app.schemas.extraction import PageExtraction, TextBlock


def normalize_block(text: str, *, kind: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00a0", " ")
    if kind in {"table", "list"}:
        return "\n".join(line.rstrip() for line in text.splitlines()).strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return " ".join(line for line in lines if line).strip()


def split_text_blocks(text: str, page_number: int = 1, *, method: str = "text") -> list[TextBlock]:
    blocks = []
    for index, part in enumerate(re.split(r"\n\s*\n", text)):
        part = part.strip()
        if not part:
            continue
        if part.startswith("#"):
            heading, _, rest = part.partition("\n")
            blocks.append(
                TextBlock(f"p{page_number}-h{index}", page_number, "heading", heading.lstrip("# "), method=method)
            )
            part = rest.strip()
            if not part:
                continue
        kind = (
            "table"
            if "|" in part and "\n" in part
            else "list" if re.match(r"^(?:[-*]|\d+[.)])\s", part) else "paragraph"
        )
        blocks.append(
            TextBlock(f"p{page_number}-b{index}", page_number, kind, normalize_block(part, kind=kind), method=method)
        )
    return blocks


def normalize_pages(pages: list[PageExtraction]) -> list[PageExtraction]:
    marginal: Counter[str] = Counter()
    for page in pages:
        marginal.update({b.text.strip() for b in page.blocks if b.bbox and (b.bbox[3] < 0.08 or b.bbox[1] > 0.92)})
    suppressed = {text for text, count in marginal.items() if count >= 3 and count / max(1, len(pages)) >= 0.6}
    result = []
    for page in pages:
        kept, warnings = [], list(page.warnings)
        for block in page.blocks:
            is_margin = block.bbox and (block.bbox[3] < 0.08 or block.bbox[1] > 0.92)
            if is_margin and block.text.strip() in suppressed:
                warnings.append("suppressed_boilerplate:" + block.block_id)
                continue
            kept.append(replace(block, text=normalize_block(block.text, kind=block.kind)))
        result.append(replace(page, blocks=kept, warnings=warnings))
    return result
