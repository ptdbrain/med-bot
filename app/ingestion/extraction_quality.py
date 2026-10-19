import re
from collections import Counter


def assess_text(text: str) -> dict[str, float]:
    length = max(1, len(text))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "characters": float(len(text.strip())),
        "words": float(len(re.findall(r"\w+", text))),
        "printable_ratio": sum(c.isprintable() or c.isspace() for c in text) / length,
        "invalid_ratio": sum(c == "\ufffd" or (not c.isprintable() and not c.isspace()) for c in text) / length,
        "repeated_line_ratio": sum(n - 1 for n in Counter(lines).values()) / max(1, len(lines)),
    }


def classify_page(text: str, *, image_coverage: float, has_visual_content: bool) -> str:
    if not text.strip():
        return "scan" if has_visual_content else "blank"
    quality = assess_text(text)
    if image_coverage >= 0.15 or quality["invalid_ratio"] > 0.02 or quality["printable_ratio"] < 0.95:
        return "mixed"
    return "text"
