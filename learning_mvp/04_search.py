from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import load_chunks, search_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 4: keyword search over saved chunks.")
    parser.add_argument("query", nargs="?", default="FAST dot quy")
    parser.add_argument("--chunks", default="learning_mvp/chunks.json")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        raise SystemExit("Run first: python learning_mvp/03_store.py") # nếu tệp chunks.json không tồn tại, chương trình sẽ dừng và yêu cầu người dùng chạy bước 3 trước để tạo tệp chunks.json

    results = search_chunks(args.query, load_chunks(chunks_path), top_k=args.top_k) # tìm kiếm các chunk phù hợp nhất với truy vấn được cung cấp, sử dụng hàm search_chunks và trả về top_k kết quả hàng đầu
    print(f"query: {args.query}")
    print(f"found {len(results)} chunks\n")
    for result in results:
        print(f"{result['chunk_id']} | score={result['score']} | source={result['source_path']}")
        print(result["text"][:260])
        print("-" * 60)


if __name__ == "__main__":
    main()
