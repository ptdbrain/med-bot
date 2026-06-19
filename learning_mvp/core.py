from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DISCLAIMER = "Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ."
MISSING_CONTEXT = "Tôi chưa tìm thấy thông tin này trong tài liệu đã được cung cấp."
STOPWORDS = {
    "ai",
    "bao",
    "bi",
    "cac",
    "cai",
    "cho",
    "co",
    "cua",
    "duoc",
    "gi",
    "hay",
    "la",
    "lieu",
    "mot",
    "nao",
    "nay",
    "noi",
    "nhung",
    "tai",
    "the",
    "thong",
    "tin",
    "toi",
    "trong",
    "ve",
}


def read_document(path: str | Path) -> dict[str, Any]:
    """Step 1: read a text/markdown file into one document dictionary."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    return {
        "title": source.stem,
        "source_path": str(source),
        "text": text,
    }


def chunk_document(document: dict[str, Any], max_chars: int = 500) -> list[dict[str, Any]]:
    """Step 2: split one document into small chunks.

    This beginner version chunks by paragraphs first. If one paragraph is too
    long, it falls back to fixed-size character chunks.
    """

    paragraphs = [part.strip() for part in document["text"].split("\n\n") if part.strip()]
    chunks: list[dict[str, Any]] = []
    buffer = ""

    def push(text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        chunk_index = len(chunks)
        title = str(document["title"])
        chunks.append(
            {
                "chunk_id": f"{title}-{chunk_index}",
                "chunk_index": chunk_index,
                "text": cleaned,
                "source_path": document["source_path"],
                "title": title,
            }
        )

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if buffer:
                push(buffer)
                buffer = ""
            for start in range(0, len(paragraph), max_chars):
                push(paragraph[start : start + max_chars])
            continue

        candidate = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(candidate) <= max_chars:
            buffer = candidate
        else:
            push(buffer)
            buffer = paragraph

    if buffer:
        push(buffer)
    return chunks


def save_chunks(chunks: list[dict[str, Any]], path: str | Path) -> None:
    """Step 3: save chunks to a JSON file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def load_chunks(path: str | Path) -> list[dict[str, Any]]:
    """Step 3: load chunks back from JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_index(input_dir: str | Path, max_chars: int = 500) -> list[dict[str, Any]]:
    """Step 6: read and chunk every .md/.txt file in one folder."""

    root = Path(input_dir)
    chunks: list[dict[str, Any]] = []
    for file_index, path in enumerate(sorted(root.iterdir())):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        document = read_document(path)
        document_chunks = chunk_document(document, max_chars=max_chars)
        for chunk in document_chunks:
            chunk["chunk_id"] = f"doc{file_index}-{chunk['chunk_id']}"
        chunks.extend(document_chunks)
    return chunks


def search_chunks(query: str, chunks: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Step 4: search chunks with a simple keyword overlap score."""

    query_tokens = set(_tokenize(query))
    results: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_tokens = _tokenize(chunk["text"])
        score = sum(1 for token in chunk_tokens if token in query_tokens)
        if score > 0:
            result = dict(chunk)
            result["score"] = score
            results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def search_chunks_tfidf(query: str, chunks: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    """Step 7: rank chunks with a tiny TF-IDF implementation.

    TF-IDF gives more weight to terms that are rare across the whole index.
    That makes specific terms like "hemoglobin" stronger than common words.
    """

    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return []

    tokenized_chunks = [_tokenize(chunk["text"]) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        document_frequency.update(set(tokens))

    total_chunks = len(chunks)
    results: list[dict[str, Any]] = []
    for chunk, tokens in zip(chunks, tokenized_chunks):
        counts = Counter(tokens)
        score = 0.0
        for token in query_tokens:
            if counts[token] == 0:
                continue
            tf = counts[token] / max(1, len(tokens))
            idf = math.log((1 + total_chunks) / (1 + document_frequency[token])) + 1
            score += tf * idf
        if score > 0:
            result = dict(chunk)
            result["score"] = round(score, 6)
            results.append(result)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def answer_from_chunks(question: str, chunks: list[dict[str, Any]]) -> str:
    """Step 5: build a tiny mock answer from retrieved chunks."""

    if not chunks:
        return f"{MISSING_CONTEXT}\n\nCảnh báo y tế: {DISCLAIMER}"

    lines = [
        "Câu hỏi:",
        question,
        "",
        "Câu trả lời học tập:",
        "Dựa trên các chunk tìm được, bạn nên đọc các đoạn liên quan bên dưới.",
        "",
        "Context dùng để trả lời:",
    ]
    for index, chunk in enumerate(chunks, start=1):
        snippet = " ".join(chunk["text"].split())
        lines.append(f"[{index}] {snippet}")

    lines.extend(["", "Nguồn:"])
    for index, chunk in enumerate(chunks, start=1):
        lines.append(f"[{index}] {chunk['source_path']} | chunk_id={chunk['chunk_id']} | score={chunk.get('score', 0)}")

    lines.extend(["", f"Cảnh báo y tế: {DISCLAIMER}"])
    return "\n".join(lines)


def _tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFD", text.lower())
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d")
    return [token for token in re.findall(r"[a-z0-9_]+", normalized) if token not in STOPWORDS]
