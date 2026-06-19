from __future__ import annotations

import ipaddress
import urllib.request
from urllib.parse import urlparse


def validate_public_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ValueError("URL must be a public HTTP or HTTPS source.")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise ValueError("URL must be a public HTTP or HTTPS source.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return url
    if not address.is_global:
        raise ValueError("URL must be a public HTTP or HTTPS source.")
    return url


class PublicHtmlCrawler:
    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout

    def fetch(self, url: str) -> bytes:
        safe_url = validate_public_url(url)
        request = urllib.request.Request(
            safe_url,
            headers={"User-Agent": "MedTutor-RAG/0.1"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return response.read()


def fetch_public_html(url: str, timeout: int = 15) -> bytes:
    return PublicHtmlCrawler(timeout=timeout).fetch(url)
