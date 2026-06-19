from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import load_chunks, search_chunks_tfidf


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 7: TF-IDF search over saved chunks.")
    parser.add_argument("query", nargs="?", default="hemoglobin xét nghiệm")
    parser.add_argument("--chunks", default="learning_mvp/index.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit("Run first: python learning_mvp/06_build_index.py")

    results = search_chunks_tfidf(args.query, load_chunks(chunks_path), top_k=args.top_k)
    print(f"query: {args.query}")
    print(f"found {len(results)} chunks with TF-IDF\n")
    for result in results:
        print(f"{result['chunk_id']} | tfidf_score={result['score']:.4f} | source={result['source_path']}")
        print(result["text"][:260])
        print("-" * 60)


if __name__ == "__main__":
    main()
