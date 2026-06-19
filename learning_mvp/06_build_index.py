from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import build_index, save_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 6: build one JSON index from many documents.")
    parser.add_argument("--input-dir", default="data/sample")
    parser.add_argument("--out", default="learning_mvp/index.json")
    parser.add_argument("--max-chars", type=int, default=350)
    args = parser.parse_args()

    chunks = build_index(Path(args.input_dir), max_chars=args.max_chars) # xây dựng một chỉ mục từ tất cả các tài liệu trong thư mục đầu vào được chỉ định, chia chúng thành các chunk dựa trên số lượng ký tự tối đa được chỉ định và trả về một danh sách các chunk
    save_chunks(chunks, Path(args.out))

    print(f"indexed {len(chunks)} chunks from {args.input_dir}") # in ra thông báo cho biết số lượng chunk đã được lập chỉ mục và thư mục đầu vào nơi chúng đã được lấy
    print(f"saved index to {args.out}")
    for chunk in chunks[:8]:
        print(f"- {chunk['chunk_id']} | {chunk['source_path']}") # in ra thông tin về mỗi chunk trong chỉ mục, bao gồm ID của chunk và đường dẫn nguồn của tài liệu mà chunk đó được tạo ra, chỉ hiển thị 8 chunk đầu tiên


if __name__ == "__main__":
    main()
