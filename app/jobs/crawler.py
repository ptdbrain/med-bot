from __future__ import annotations

import urllib.request


def fetch_public_html(url: str, timeout: int = 15) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()
