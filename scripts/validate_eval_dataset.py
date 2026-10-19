import argparse
import json
from pathlib import Path

from app.evaluation.dataset import load_cases, validate_cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=Path("evaluation/corpus/manifest.json"))
    parser.add_argument("--cases", type=Path, default=Path("evaluation/cases"))
    parser.add_argument("--require-review", action="store_true")
    args = parser.parse_args()
    paths = sorted(args.cases.glob("*.jsonl")) if args.cases.is_dir() else [args.cases]
    cases = [case for path in paths for case in load_cases(path)]
    errors = validate_cases(
        cases, json.loads(args.corpus.read_text(encoding="utf-8")), require_review=args.require_review
    )
    if not cases:
        errors.append("empty_dataset")
    print(json.dumps({"cases": len(cases), "reviewed": sum(bool(c.reviewed_by) for c in cases), "errors": errors}))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
