from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from learning_mvp.core import read_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: read a .txt/.md file.") #đọc một file .txt/.md
    parser.add_argument("path", nargs="?", default="data/sample/stroke.md") # đường dẫn đến file cần đọc, mặc định là "data/sample/stroke.md"
    args = parser.parse_args()  # phân tích các đối số dòng lệnh

    document = read_document(Path(args.path))
    print("DOCUMENT")
    print(f"title: {document['title']}")
    print(f"source_path: {document['source_path']}")
    print("\nTEXT PREVIEW")
    print(document["text"][:700])


if __name__ == "__main__":
    main()
