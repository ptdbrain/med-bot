from dataclasses import replace

import pytest

from app import config


def test_profile_keeps_explicit_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    settings = config.resolve_profile("local", {"VECTOR_STORE_PROVIDER": "sqlite-dense"})
    assert settings.vector_store_provider == "sqlite-dense"
    assert settings.llm_provider == "ollama"
    assert settings.profile == "local"


def test_profile_rejects_invalid_values():
    with pytest.raises(ValueError):
        config.resolve_profile("unknown", {})
    with pytest.raises(ValueError):
        config.resolve_profile("demo", {"EMBEDDING_BATCH_SIZE": "0"})


def test_demo_doctor_does_not_require_model_downloads(tmp_path):
    from app.runtime.doctor import check_runtime

    settings = replace(config.resolve_profile("demo", {}), data_dir=tmp_path)
    checks = check_runtime(settings)
    assert not any(c.status in {"error", "missing"} for c in checks)


def test_ollama_unreachable_is_not_ready(tmp_path):
    from app.runtime.doctor import check_runtime

    settings = replace(
        config.resolve_profile("demo", {}),
        llm_provider="ollama",
        ollama_base_url="http://127.0.0.1:1",
        data_dir=tmp_path,
    )
    checks = check_runtime(settings, probe_models=True)
    assert any(c.name == "generator" and c.status == "missing" for c in checks)
    assert all("api_key" not in c.message for c in checks)
