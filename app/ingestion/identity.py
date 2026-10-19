from __future__ import annotations

import hashlib
import json
import unicodedata


def identity_hash(parts: list[str]) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def document_id(collection_id: str, source_key: str) -> str:
    key = unicodedata.normalize("NFC", source_key.replace("\\", "/"))
    return "doc-" + identity_hash([collection_id, key])[:24]


def revision_id(content: bytes, extraction_fingerprint: str) -> str:
    return "rev-" + identity_hash([hashlib.sha256(content).hexdigest(), extraction_fingerprint])[:24]
