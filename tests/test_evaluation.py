from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.evaluation.answer_eval import (
    evaluate_answers,
    has_required_answer_format,
)
from app.main import create_app
from app.safety.policy import MEDICAL_DISCLAIMER


class EvaluationTests(unittest.TestCase):
    def test_eval_api_returns_metrics_and_auditable_case_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = TestClient(
                create_app(
                    Settings(
                        data_dir=root,
                        index_path=root / "indexes" / "vectors.json",
                        metadata_db_path=root / "indexes" / "metadata.db",
                    )
                )
            )
            client.post(
                "/ingest/file",
                files={
                    "file": (
                        "stroke.md",
                        b"# FAST\n\nFAST gom Face, Arm, Speech va Time.",
                        "text/markdown",
                    )
                },
            )
            client.post(
                "/ingest/file",
                files={
                    "file": (
                        "hypertension.md",
                        b"# Tang huyet ap\n\nTang huyet ap la yeu to nguy co tim mach.",
                        "text/markdown",
                    )
                },
            )

            response = client.post(
                "/eval/run",
                json={
                    "dataset_path": "app/evaluation/eval_dataset.json",
                    "top_k": 5,
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        metric_names = {metric["name"] for metric in payload["metrics"]}
        self.assertTrue(
            {
                "hit_at_k",
                "mrr",
                "citation_rate",
                "format_compliance_rate",
                "disclaimer_rate",
                "insufficient_context_accuracy",
                "safety_accuracy",
            }.issubset(metric_names)
        )
        metrics = {metric["name"]: metric["value"] for metric in payload["metrics"]}
        self.assertEqual(metrics["citation_rate"], 1.0)
        self.assertEqual(metrics["format_compliance_rate"], 1.0)
        self.assertEqual(metrics["insufficient_context_accuracy"], 1.0)
        self.assertTrue(payload["details"])
        categories = {detail["category"] for detail in payload["details"]}
        self.assertEqual(categories, {"retrieval", "answer", "safety"})
        self.assertTrue(all("question" in detail and "passed" in detail for detail in payload["details"]))

    def test_eval_details_include_actual_safety_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = TestClient(
                create_app(
                    Settings(
                        data_dir=root,
                        index_path=root / "indexes" / "vectors.json",
                        metadata_db_path=root / "indexes" / "metadata.db",
                    )
                )
            )

            payload = client.post("/eval/run", json={"top_k": 3}).json()

        emergency = next(
            detail
            for detail in payload["details"]
            if detail.get("category") == "safety" and detail.get("expected") == "emergency"
        )
        self.assertEqual(emergency["actual_warning"], "emergency")
        self.assertTrue(emergency["passed"])

    def test_answer_eval_requires_literal_citation_and_required_sections(
        self,
    ) -> None:
        class SourceOnlyService:
            def chat(self, question: str, rag_mode: str):
                return SimpleNamespace(
                    answer=f"Grounded answer without citation.\n{MEDICAL_DISCLAIMER}",
                    sources=[{"chunk_id": "source-1"}],
                )

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "eval.json"
            dataset.write_text(
                '[{"question":"Q","expected":"answerable"}]',
                encoding="utf-8",
            )
            metrics = {
                metric.name: metric.value
                for metric in evaluate_answers(
                    SourceOnlyService(),
                    str(dataset),
                )
            }

        self.assertEqual(metrics["citation_rate"], 0.0)
        self.assertEqual(metrics["format_compliance_rate"], 0.0)

    def test_answer_format_accepts_four_numbered_blocks(self) -> None:
        answer = "\n\n".join(f"{index}. Nội dung phần {index}" for index in range(1, 5))

        self.assertTrue(has_required_answer_format(answer))
        self.assertFalse(
            has_required_answer_format("1. Chỉ có một phần"),
        )


if __name__ == "__main__":
    unittest.main()
