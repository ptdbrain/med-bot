from __future__ import annotations

from pathlib import Path


def main() -> None:
    for path in [Path("data/indexes/vector_store.json"), Path("data/indexes/metadata.db")]:
        if path.exists():
            path.unlink()
            print(f"removed {path}")


if __name__ == "__main__":
    main()
