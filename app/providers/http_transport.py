import time

import requests


def is_retryable_status(status: int) -> bool:
    return status in {429, 502, 503, 504}


class HTTPTransport:
    def __init__(self, settings, session=None):
        if not settings.http_provider_url or not settings.http_provider_key:
            raise ValueError("HTTP provider requires endpoint and credentials")
        self.url = settings.http_provider_url.rstrip("/")
        self.key, self.timeout = settings.http_provider_key, settings.http_provider_timeout
        self.session = session or requests.Session()

    def post(self, path: str, payload: dict) -> dict:
        deadline = time.monotonic() + self.timeout
        for attempt in range(3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("HTTP provider deadline exceeded")
            try:
                response = self.session.post(
                    self.url + path,
                    json=payload,
                    headers={"Authorization": "Bearer " + self.key},
                    timeout=(min(5, remaining), remaining),
                )
            except requests.RequestException as exc:
                raise RuntimeError("HTTP provider connection failed") from exc
            if is_retryable_status(response.status_code) and attempt < 2:
                try:
                    wait = min(2.0, max(0, float(response.headers.get("Retry-After", attempt + 0.5))))
                except ValueError:
                    wait = 0.5
                time.sleep(min(wait, max(0, deadline - time.monotonic())))
                continue
            if not 200 <= response.status_code < 300:
                raise RuntimeError(f"HTTP provider returned status {response.status_code}")
            try:
                body = response.json()
                if not isinstance(body, dict):
                    raise TypeError("Response must be an object")
                return body
            except (TypeError, ValueError) as exc:
                raise RuntimeError("HTTP provider returned invalid JSON") from exc
        raise RuntimeError("HTTP provider retry budget exhausted")
