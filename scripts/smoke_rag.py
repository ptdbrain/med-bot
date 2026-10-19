from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import requests


def check_answer_payload(payload: dict) -> list[str]:
    errors = []
    answer = payload.get("answer")
    sources = payload.get("sources")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("empty_answer")
        return errors
    if not isinstance(sources, list) or not sources:
        errors.append("missing_evidence")
    if answer.strip().lower().startswith(("reasoning:", "let me think", "analysis:")):
        errors.append("reasoning_only")
    source_ids = {str(source.get("chunk_id")) for source in sources or [] if isinstance(source, dict)}
    cited = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    if source_ids and not cited:
        errors.append("missing_citation")
    if cited and any(value < 1 or value > len(source_ids) for value in cited):
        errors.append("invalid_citation_id")
    return errors


def run_smoke(base_url: str, fixture_dir: Path, output: Path) -> dict:
    session = requests.Session()
    report = {"base_url": base_url.rstrip("/"), "checks": [], "errors": []}
    for endpoint in ("/health", "/ready"):
        try:
            response = session.get(report["base_url"] + endpoint, timeout=(3, 10))
            report["checks"].append({"endpoint": endpoint, "status": response.status_code})
            if response.status_code != 200:
                report["errors"].append(f"{endpoint}:http_{response.status_code}")
        except requests.RequestException as exc:
            report["errors"].append(f"{endpoint}:{type(exc).__name__}")
            break
    for path in sorted(fixture_dir.glob("*.pdf")):
        try:
            with path.open("rb") as stream:
                response = session.post(
                    report["base_url"] + "/ingest",
                    files={"file": (path.name, stream, "application/pdf")},
                    timeout=(3, 180),
                )
            report["checks"].append({"fixture": path.name, "status": response.status_code})
            if response.status_code not in (200, 202):
                report["errors"].append(f"{path.name}:http_{response.status_code}")
        except requests.RequestException as exc:
            report["errors"].append(f"{path.name}:{type(exc).__name__}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/reports/smoke.json"))
    args = parser.parse_args()
    report = run_smoke(args.base_url, args.fixture_dir, args.output)
    print(json.dumps({"checks": len(report["checks"]), "errors": report["errors"]}, ensure_ascii=False))
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
