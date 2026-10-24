from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

import pytest

from app.config import Settings, get_settings, resolve_profile
from app.runtime import doctor
from scripts.doctor import main


def test_doctor_plain_text_output_does_not_crash(capsys):
    with patch("scripts.doctor.check_runtime", return_value=[doctor.RuntimeCheck("probe", "ok", "verified")]):
        assert main(["--profile", "demo"]) == 0
    assert "OK" in capsys.readouterr().out


def test_application_reads_same_profile_as_doctor(monkeypatch):
    for key in ("LLM_PROVIDER", "EMBEDDING_PROVIDER", "VECTOR_STORE_PROVIDER", "RERANKER_PROVIDER"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_PROFILE", "local")
    settings = get_settings()
    assert settings.profile_name == "local"
    assert settings.llm_provider == "ollama"
    assert settings.embedding_provider == "sentence-transformers"


def test_data_directory_override_isolates_all_index_paths(tmp_path):
    settings = resolve_profile("local", {"DATA_DIR": str(tmp_path)})
    assert settings.data_dir == tmp_path
    assert settings.metadata_db_path.is_relative_to(tmp_path)
    assert settings.dense_index_path.is_relative_to(tmp_path)
    assert settings.index_path.is_relative_to(tmp_path)


def test_unknown_override_is_not_silently_ignored():
    with pytest.raises(ValueError, match="override"):
        resolve_profile("local", {"VECTRO_STORE_PROVIDER": "memory"})


def test_invalid_ollama_json_is_structured_error():
    response = type("InvalidResponse", (), {"status_code": 200, "json": lambda self: []})()
    with patch.object(doctor.requests, "get", return_value=response):
        checks = doctor.check_runtime(Settings(llm_provider="ollama"))
    assert not doctor.is_ready(checks)
    assert any(check.status == "error" for check in checks)


def test_qdrant_named_vectors_not_accepted_by_unnamed_adapter():
    def response(payload):
        return type("Response", (), {"status_code": 200, "json": lambda self: payload})()

    collection = {"result": {"config": {"params": {"vectors": {"named": {"size": 1024}}}}}}
    with patch.object(doctor.requests, "get", side_effect=[response({}), response(collection)]):
        checks = doctor.check_runtime(Settings(vector_store_provider="qdrant"))
    assert not doctor.is_ready(checks)


def test_empty_local_directory_is_not_a_cached_model(tmp_path):
    assert doctor._cached_model_path(str(tmp_path)) is None


def test_unavailable_model_probe_has_no_hidden_download(tmp_path):
    settings = replace(Settings(), embedding_provider="sentence-transformers", embedding_model=str(tmp_path))
    with patch.object(doctor, "_package_available", return_value=True):
        checks = doctor.check_runtime(settings, probe_models=True)
    assert next(check for check in checks if check.name == "embedding_model").status == "missing"


def test_json_report_reports_not_ready_without_runtime(capsys):
    with patch("scripts.doctor.check_runtime", return_value=[doctor.RuntimeCheck("ollama", "missing", "offline")]):
        assert main(["--profile", "local", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False


def test_doctor_can_save_reproducible_json_report(tmp_path, capsys):
    path = tmp_path / "reports" / "runtime.json"
    with patch("scripts.doctor.check_runtime", return_value=[doctor.RuntimeCheck("probe", "ok", "verified")]):
        assert main(["--output", str(path), "--json"]) == 0
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report == json.loads(capsys.readouterr().out)
    assert report["probe_models"] is False


def test_dense_store_rejects_lexical_embedding_configuration():
    checks = doctor.check_runtime(Settings(vector_store_provider="sqlite-dense", embedding_provider="lexical"))
    assert not doctor.is_ready(checks)


def test_dense_embedding_with_lexical_memory_store_cannot_be_reported_ready():
    with patch.object(doctor, "_check_embedding", return_value=doctor.RuntimeCheck("embedding_model", "ok", "cached")):
        checks = doctor.check_runtime(
            Settings(vector_store_provider="memory", embedding_provider="sentence-transformers")
        )
    assert not doctor.is_ready(checks)


def test_doctor_default_profile_matches_api_environment(monkeypatch, capsys):
    monkeypatch.setenv("RAG_PROFILE", "local")
    with patch("scripts.doctor.check_runtime", return_value=[]) as checks:
        assert main(["--json"]) == 0
    assert checks.call_args.args[0].profile_name == "local"
    assert json.loads(capsys.readouterr().out)["profile"] == "local"


@pytest.mark.parametrize("generated", [None, 123, True, {}, []])
def test_nonstring_generation_is_not_successful_inference(generated):
    from unittest.mock import Mock

    tags = Mock(status_code=200)
    tags.json.return_value = {"models": [{"name": Settings().ollama_model}]}
    response = Mock(status_code=200)
    response.json.return_value = {"response": generated}
    with (
        patch.object(doctor.requests, "get", return_value=tags),
        patch.object(doctor.requests, "post", return_value=response),
    ):
        checks = doctor.check_runtime(Settings(llm_provider="ollama"), probe_models=True)
    assert next(check for check in checks if check.name == "ollama_inference").status == "error"


def test_hf_hub_cache_takes_precedence_over_legacy_cache(monkeypatch, tmp_path):
    current = tmp_path / "hub" / "models--example--model" / "snapshots" / "revision"
    current.mkdir(parents=True)
    (current / "config.json").write_text("{}", encoding="utf-8")
    (current / "model.safetensors").write_bytes(b"test-weight-placeholder")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "legacy"))
    assert doctor._cached_model_path("example/model") == current
