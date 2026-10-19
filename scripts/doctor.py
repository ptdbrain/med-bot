from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

from app.config import get_settings
from app.runtime.doctor import check_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Check providers without downloading models.")
    parser.add_argument("--profile", choices=["demo", "local", "cloud", "hybrid"])
    parser.add_argument("--env-file")
    parser.add_argument("--probe-models", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    if args.profile:
        os.environ["RAG_PROFILE"] = args.profile
    checks = check_runtime(get_settings(), probe_models=args.probe_models)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=True))
    else:
        for check in checks:
            print(f"{check.name}: {check.status} - {check.message}")
    return int(any(check.status != "ok" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
