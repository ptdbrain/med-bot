from __future__ import annotations

import json
from pathlib import Path

from app.evaluation.benchmark import run_benchmark
from app.evaluation.dataset import load_cases, validate_cases


def test_reviewed_dataset_contract_and_provenance():
    manifest_path = Path("evaluation/reviewed_education/corpus/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == "reviewed-education-v1"
    assert manifest["provenance"] == "clinically_reviewed_educational_v1"

    dev_cases = load_cases(Path("evaluation/reviewed_education/cases/dev.jsonl"))
    test_cases = load_cases(Path("evaluation/reviewed_education/cases/test.jsonl"))
    all_cases = dev_cases + test_cases

    # 1. No unreviewed cases
    assert all(c.reviewed_by is not None for c in all_cases)
    assert all(c.provenance == "reviewed_clinical_v1" for c in all_cases)

    # 2. Strict validation passes require_review=True with zero errors
    errors = validate_cases(all_cases, manifest, require_review=True)
    assert errors == []


def test_reviewed_benchmark_passes_quality_gate_without_allow_draft(tmp_path: Path):
    config_path = tmp_path / "reviewed_config.json"
    output_path = tmp_path / "reviewed_report.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "demo",
                "overrides": {
                    "DATA_DIR": str(tmp_path / "data_reviewed"),
                    "VECTOR_STORE_PROVIDER": "memory",
                },
                "corpus": "evaluation/reviewed_education/corpus/manifest.json",
                "cases": "evaluation/reviewed_education/cases",
                "modes": ["bm25", "lexical_vector"],
                "generate": False,
            }
        ),
        encoding="utf-8",
    )

    # Must pass WITHOUT allow_draft because all cases have reviewed_by set
    report = run_benchmark(config_path, "dev", output_path, allow_draft=False)
    assert report["quality_gate"] == "passed_retrieval_quality_gate"
    assert report["dataset_reviewed"] is True
    assert report["summary"]["bm25"]["recall_at_5"] == 1.0
