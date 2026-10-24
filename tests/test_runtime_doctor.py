from __future__ import annotations

import os
import unittest
from importlib import import_module
from unittest.mock import patch

from app import config
from app.config import Settings

try:
    doctor = import_module("app.runtime.doctor")
except ImportError:
    doctor = None


def _resolve_profile(name: str, overrides: dict[str, str] | None = None):
    resolver = getattr(config, "resolve_profile", None)
    if resolver is None:
        raise AssertionError("app.config.resolve_profile is not implemented")
    return resolver(name, overrides)


def _doctor():
    if doctor is None:
        raise AssertionError("app.runtime.doctor is not implemented")
    return doctor


class RuntimeDoctorTests(unittest.TestCase):
    def test_explicit_provider_override_wins_over_local_profile(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = _resolve_profile("local", {"VECTOR_STORE_PROVIDER": "sqlite-dense"})

        self.assertEqual(settings.profile_name, "local")
        self.assertEqual(settings.embedding_provider, "sentence-transformers")
        self.assertEqual(settings.vector_store_provider, "sqlite-dense")

    def test_environment_override_wins_over_profile_defaults(self) -> None:
        with patch.dict(os.environ, {"VECTOR_STORE_PROVIDER": "sqlite-dense"}, clear=True):
            settings = _resolve_profile("local")

        self.assertEqual(settings.vector_store_provider, "sqlite-dense")

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile"):
            _resolve_profile("unknown")

    def test_local_profile_is_not_ready_when_services_are_unreachable(self) -> None:
        settings = _resolve_profile("local", {})
        runtime_doctor = _doctor()
        with patch.object(runtime_doctor.requests, "get", side_effect=runtime_doctor.requests.ConnectionError):
            checks = runtime_doctor.check_runtime(settings)

        self.assertFalse(runtime_doctor.is_ready(checks))
        statuses = {check.name: check.status for check in checks}
        self.assertEqual(statuses["ollama"], "missing")
        self.assertEqual(statuses["qdrant"], "missing")

    def test_doctor_reports_missing_model_dependency_without_loading_or_download(self) -> None:
        settings = Settings(
            llm_provider="mock",
            embedding_provider="sentence-transformers",
            vector_store_provider="sqlite-dense",
            reranker_provider="cross-encoder",
        )
        runtime_doctor = _doctor()
        with (
            patch.object(runtime_doctor, "_package_available", return_value=False),
            patch.object(runtime_doctor, "_cached_model_path", return_value=None),
        ):
            checks = runtime_doctor.check_runtime(settings, probe_models=False)

        statuses = {check.name: check.status for check in checks}
        self.assertEqual(statuses["embedding_model"], "missing")
        self.assertEqual(statuses["reranker_model"], "missing")

    def test_local_profile_reports_pdf_ocr_gap(self) -> None:
        settings = _resolve_profile("local", {})
        runtime_doctor = _doctor()
        with patch.object(runtime_doctor.requests, "get", side_effect=runtime_doctor.requests.ConnectionError):
            checks = runtime_doctor.check_runtime(settings)

        ocr = next(check for check in checks if check.name == "pdf_ocr")
        self.assertEqual(ocr.status, "missing")

    def test_readiness_ignores_optional_not_checked_components(self) -> None:
        runtime_doctor = _doctor()

        self.assertTrue(
            runtime_doctor.is_ready([runtime_doctor.RuntimeCheck("reranker", "not_checked", "lexical mode")])
        )
        self.assertFalse(runtime_doctor.is_ready([runtime_doctor.RuntimeCheck("qdrant", "error", "unavailable")]))


if __name__ == "__main__":
    unittest.main()
