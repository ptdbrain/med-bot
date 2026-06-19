from __future__ import annotations

import os
import unittest
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
            {
                "FRONTEND_ORIGINS": (
                    " https://medtutor.example.com, http://localhost:3000, ,"
                )
            },
            clear=True,
        ):
            settings = get_settings()

        self.assertEqual(
            settings.frontend_origins,
            ("https://medtutor.example.com", "http://localhost:3000"),
        )


if __name__ == "__main__":
    unittest.main()
