from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import chunk_document, read_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: split a document into chunks.") # chia tài liệu thành các chunk
    parser.add_argument("path", nargs="?", default="data/sample/stroke.md") # đường dẫn đến file cần đọc, mặc định là "data/sample/stroke.md"
    parser.add_argument("--max-chars", type=int, default=350)   # số lượng ký tự tối đa cho mỗi chunk, mặc định là 350
    args = parser.parse_args() # phân tích các đối số dòng lệnh

    document = read_document(Path(args.path)) # đọc tài liệu từ đường dẫn được cung cấp
    chunks = chunk_document(document, max_chars=args.max_chars) # chia tài liệu thành các chunk dựa trên số lượng ký tự tối đa được chỉ định

    print(f"created {len(chunks)} chunks\n")
    for chunk in chunks: # in ra thông tin về mỗi chunk, bao gồm ID của chunk, số lượng ký tự trong chunk và một phần xem trước của văn bản trong chunk
        print(f"{chunk['chunk_id']} | chars={len(chunk['text'])}")  
        print(chunk["text"][:220])
        print("-" * 60)


if __name__ == "__main__":
    main()
