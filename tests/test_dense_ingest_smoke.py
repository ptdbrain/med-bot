from __future__ import annotations

import importlib
import json
import math

import pytest


class FixtureEmbedding:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if "SMOKE-2026" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


def runner():
    try:
        return importlib.import_module("scripts.smoke_dense_ingest").run_smoke
    except ModuleNotFoundError:
        pytest.fail("scripts.smoke_dense_ingest is not implemented")


def test_smoke_persists_vectors_and_retrieves_expected_evidence_after_reopen(tmp_path):
    root = tmp_path / "run"
    report = runner()(root, embedding_provider=FixtureEmbedding(), dimension=2)
    assert report["passed"]
    assert report["documents"] == 2
    assert report["index_check"]["passed"]
    assert report["index_check"]["vectors"] > 0
    assert report["vector_validation"]["passed"]
    assert all(case["passed"] for case in report["queries"])
    assert all("SMOKE-2026" in case["evidence"] for case in report["queries"])
    assert report["embedding_execution"] == "injected_test_provider"
    assert report["generation"] == "not_run"
    assert report["active_revision"] == "not_implemented"
    assert json.loads((root / "report.json").read_text(encoding="utf-8"))["passed"]


def test_existing_directory_is_never_overwritten(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    marker = root / "existing.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner()(root, embedding_provider=FixtureEmbedding(), dimension=2)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_failure_is_saved_in_report_without_claiming_pass(tmp_path):
    class FailingEmbedding(FixtureEmbedding):
        def embed_documents(self, texts):
            raise RuntimeError("fixture failure")

    root = tmp_path / "failure"
    report = runner()(root, embedding_provider=FailingEmbedding(), dimension=2)
    assert not report["passed"]
    assert report["errors"]
    assert (root / "report.json").is_file()


def test_zero_vectors_fail_even_with_matching_index_ids(tmp_path):
    class ZeroEmbedding(FixtureEmbedding):
        def embed_documents(self, texts):
            return [[0.0, 0.0] for _ in texts]

    report = runner()(tmp_path / "zeros", embedding_provider=ZeroEmbedding(), dimension=2)
    assert not report["passed"]
    assert not report["vector_validation"]["passed"]


def test_nonfinite_vectors_fail_validation(tmp_path):
    class NonfiniteEmbedding(FixtureEmbedding):
        def embed_documents(self, texts):
            return [[math.nan, 1.0] for _ in texts]

    report = runner()(tmp_path / "nan", embedding_provider=NonfiniteEmbedding(), dimension=2)
    assert not report["passed"]
