from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.config import resolve_profile
from app.runtime.doctor import check_runtime, is_ready


def _parse_override(value: str) -> tuple[str, str]:
    key, separator, setting = value.partition("=")
    if not separator or not key.strip():
        raise argparse.ArgumentTypeError("Overrides must use KEY=VALUE format.")
    return key.strip(), setting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe configured RAG runtime without downloading models")
    parser.add_argument("--profile", choices=("demo", "local"), default=os.getenv("RAG_PROFILE", "demo"))
    parser.add_argument("--set", dest="overrides", action="append", type=_parse_override, default=[])
    parser.add_argument("--probe-models", action="store_true", help="Run local embedding, reranker, and LLM inference")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", type=Path, help="Save the same JSON report as an artifact")
    args = parser.parse_args(argv)
    try:
        settings = resolve_profile(args.profile, dict(args.overrides))
    except ValueError as exc:
        parser.error(str(exc))
    checks = check_runtime(settings, probe_models=args.probe_models)
    ready = is_ready(checks)
    report = {
        "profile": settings.profile_name,
        "ready": ready,
        "probe_models": args.probe_models,
        "checked_at": datetime.now(UTC).isoformat(),
        "checks": [{"name": check.name, "status": check.status, "message": check.message} for check in checks],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Profile {report['profile']}: {'READY' if ready else 'NOT READY'}")
        for check in checks:
            print(f"{check.status.upper():12} {check.name}: {check.message}")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
