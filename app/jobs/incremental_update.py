from __future__ import annotations

from app.utils.hashing import stable_hash


def has_changed(previous_hash: str | None, content: bytes) -> tuple[bool, str]:
    current_hash = stable_hash(content.decode("utf-8", errors="replace"))
    return previous_hash != current_hash, current_hash
