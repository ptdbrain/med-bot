import argparse
import json
from pathlib import Path

from app.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/benchmarks/rag-demo-regression.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--output", type=Path, default=Path("data/reports/rag-dev.json"))
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = run_benchmark(args.config, args.split, args.output, allow_draft=args.allow_draft, limit=args.limit)
    print(
        json.dumps(
            {"quality_gate": report["quality_gate"], "case_count": report["case_count"], "summary": report["summary"]},
            indent=2,
        )
    )
    return int(any(m["failures"] for m in report["summary"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
