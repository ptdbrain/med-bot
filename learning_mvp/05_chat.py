from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import answer_from_chunks, load_chunks, search_chunks, search_chunks_tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 5: mini RAG = question -> search -> answer with sources.")
    parser.add_argument("question", nargs="?", default="FAST trong đột quỵ gồm những dấu hiệu nào?")
    parser.add_argument("--chunks", default="learning_mvp/chunks.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--search-mode", choices=["keyword", "tfidf"], default="keyword")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit("Run first: python learning_mvp/03_store.py")

    chunks = load_chunks(chunks_path)
    if args.search_mode == "tfidf":
        results = search_chunks_tfidf(args.question, chunks, top_k=args.top_k)
    else:
        results = search_chunks(args.question, chunks, top_k=args.top_k)
    print(answer_from_chunks(args.question, results))


if __name__ == "__main__":
    main()
