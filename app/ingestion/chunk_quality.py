from collections import Counter

from app.schemas.document import Chunk


def assess_chunks(chunks: list[Chunk]) -> dict:
    lengths = sorted(c.metadata.extra.get("token_count", len(c.text.split())) for c in chunks)
    hashes = Counter(" ".join(c.text.split()) for c in chunks)
    return {
        "chunks": len(chunks),
        "empty": sum(not c.text.strip() for c in chunks),
        "duplicates": sum(n - 1 for n in hashes.values()),
        "token_p50": lengths[len(lengths) // 2] if lengths else 0,
        "token_p95": lengths[min(len(lengths) - 1, int(len(lengths) * 0.95))] if lengths else 0,
        "token_max": max(lengths, default=0),
        "short_semantic_units": sum("short_semantic_unit" in c.metadata.extra.get("warnings", []) for c in chunks),
    }
