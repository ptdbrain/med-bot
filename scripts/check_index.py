from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings


def inspect_index(db_path: Path, *, read_only: bool = True) -> dict:
    if not db_path.is_file():
        return {"status": "missing", "metadata_db": str(db_path), "documents": 0, "chunks": 0}
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro" if read_only else str(db_path)
    import sqlite3

    connection = sqlite3.connect(uri, uri=read_only)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"documents", "chunks"}.issubset(tables):
            return {
                "status": "invalid",
                "metadata_db": str(db_path),
                "missing_tables": sorted({"documents", "chunks"} - tables),
            }
        docs = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        active = (
            connection.execute("SELECT COUNT(*) FROM active_revisions").fetchone()[0]
            if "active_revisions" in tables
            else None
        )
        return {
            "status": "ok",
            "metadata_db": str(db_path),
            "documents": docs,
            "chunks": chunks,
            "active_revisions": active,
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-only", action="store_true", default=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    result = inspect_index(settings.metadata_db_path, read_only=args.read_only)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print("\n".join(f"{key}: {value}" for key, value in result.items()))
    return int(result["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
