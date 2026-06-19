from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import chunk_document, read_document, save_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 3: save chunks to JSON.")
    parser.add_argument("path", nargs="?", default="data/sample/stroke.md")
    parser.add_argument("--out", default="learning_mvp/chunks.json")
    parser.add_argument("--max-chars", type=int, default=350)
    args = parser.parse_args()

    document = read_document(Path(args.path)) 
    chunks = chunk_document(document, max_chars=args.max_chars) # chia tài liệu thành các chunk dựa trên số lượng ký tự tối đa được chỉ định
    save_chunks(chunks, Path(args.out)) # lưu các chunk vào một tệp JSON tại đường dẫn được chỉ định
    print(f"saved {len(chunks)} chunks to {args.out}") # in ra thông báo cho biết số lượng chunk đã được lưu và đường dẫn đến tệp JSON nơi chúng đã được lưu


if __name__ == "__main__":
    main()
