from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config import get_settings


class SettingsTests(unittest.TestCase):
    def test_frontend_origins_default_to_local_next_app(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()

        self.assertEqual(settings.frontend_origins, ("http://localhost:3000",))

    def test_frontend_origins_are_trimmed_and_empty_values_are_removed(self) -> None:
        with patch.dict(
            os.environ,
            {"FRONTEND_ORIGINS": (" https://medtutor.example.com, http://localhost:3000, ,")},
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(
            settings.frontend_origins,
            ("https://medtutor.example.com", "http://localhost:3000"),
        )

    def test_advanced_rag_settings_are_loaded_from_environment(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CHUNK_SIZE": "1200",
                "CHUNK_OVERLAP": "180",
                "CHUNK_MIN_SIZE": "240",
                "HYBRID_DENSE_WEIGHT": "1.5",
                "HYBRID_SPARSE_WEIGHT": "0.8",
                "RERANKER_PROVIDER": "cross-encoder",
                "RERANKER_CANDIDATE_K": "30",
                "RERANKER_DIVERSITY_LAMBDA": "0.65",
                "DENSE_INDEX_PATH": "custom/dense.db",
                "DENSE_WRITE_BATCH_SIZE": "32",
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(settings.chunk_size, 1200)
        self.assertEqual(settings.chunk_overlap, 180)
        self.assertEqual(settings.chunk_min_size, 240)
        self.assertEqual(settings.hybrid_dense_weight, 1.5)
        self.assertEqual(settings.hybrid_sparse_weight, 0.8)
        self.assertEqual(settings.reranker_provider, "cross-encoder")
        self.assertEqual(settings.reranker_candidate_k, 30)
        self.assertEqual(settings.reranker_diversity_lambda, 0.65)
        self.assertEqual(settings.dense_index_path, Path("custom/dense.db"))
        self.assertEqual(settings.dense_write_batch_size, 32)


if __name__ == "__main__":
    unittest.main()


def test_dotenv_is_loaded_without_overriding_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("RAG_PROFILE=demo\nCHUNK_SIZE=777\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAG_PROFILE", raising=False)
    monkeypatch.delenv("CHUNK_SIZE", raising=False)
    settings = get_settings()
    assert settings.chunk_size == 777

    monkeypatch.setenv("CHUNK_SIZE", "888")
    assert get_settings().chunk_size == 888
