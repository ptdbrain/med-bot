import io
import json
import queue

import pytest

from app.ingestion.ocr.paddle import OCRWorkerError, PaddleOCRProvider
from app.ingestion.ocr.worker import serve


class FakeProvider:
    def recognize(self, image, *, page_number, languages, **kwargs):
        from app.schemas.extraction import OCRResult

        return OCRResult((), None, (), {"page_number": page_number, "correlation_id": kwargs["correlation_id"]})


def test_worker_serves_multiple_correlated_page_requests():
    requests = [
        {
            "request_id": "r1",
            "image": "YQ==",
            "page_number": 1,
            "languages": ["vi"],
            "correlation_id": "trace-1",
            "dpi": 160,
            "image_dimensions_before_resize": [100, 100],
            "image_dimensions_after_resize": [80, 80],
        },
        {
            "request_id": "r2",
            "image": "Yg==",
            "page_number": 2,
            "languages": ["vi"],
            "correlation_id": "trace-1",
            "dpi": 160,
            "image_dimensions_before_resize": [100, 100],
            "image_dimensions_after_resize": [80, 80],
        },
    ]
    source = io.StringIO("\n".join(json.dumps(item) for item in requests) + "\n")
    output = io.StringIO()
    providers = []

    def factory():
        provider = FakeProvider()
        providers.append(provider)
        return provider

    serve(source, output, factory)
    assert len(providers) == 1
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [reply["request_id"] for reply in replies] == ["r1", "r2"]
    assert [reply["result"]["diagnostics"]["page_number"] for reply in replies] == [1, 2]
    assert all(reply["correlation_id"] == "trace-1" for reply in replies)
    assert all(reply["result"]["diagnostics"]["dpi"] == 160 for reply in replies)
    assert all(reply["result"]["diagnostics"]["image_dimensions_after_resize"] == [80, 80] for reply in replies)
    assert all(reply["result"]["diagnostics"]["paddleocr_version"] for reply in replies)


def test_worker_returns_structured_invalid_request_error():
    output = io.StringIO()
    serve(io.StringIO("[]\n"), output, lambda: FakeProvider())
    response = json.loads(output.getvalue())
    assert response["error"]["category"] == "invalid_request"
    assert response["error"]["type"] == "TypeError"


def test_worker_marks_restart_at_configured_batch_limit():
    request = {"request_id": "r1", "correlation_id": "trace", "image": "YQ==", "page_number": 1, "languages": ["vi"]}
    output = io.StringIO()
    serve(io.StringIO(json.dumps(request) + "\n"), output, lambda: FakeProvider(), max_requests=1)
    assert json.loads(output.getvalue())["restart_required"] is True


def test_model_value_error_is_not_mislabeled_as_bad_request():
    class BrokenProvider:
        def recognize(self, *args, **kwargs):
            raise ValueError("model configuration mismatch")

    request = {"request_id": "r1", "correlation_id": "trace", "image": "YQ==", "page_number": 1, "languages": ["vi"]}
    output = io.StringIO()
    serve(io.StringIO(json.dumps(request) + "\n"), output, BrokenProvider)
    response = json.loads(output.getvalue())
    assert response["error"]["category"] == "provider_error"
    assert response["error"]["message"] == "model configuration mismatch"


class QueuePipe:
    def __init__(self, on_write=None, initial=()):
        self.items = queue.Queue()
        self.on_write = on_write
        for item in initial:
            self.items.put(item)

    def write(self, value):
        self.on_write(value)

    def flush(self):
        pass

    def close(self):
        self.items.put(None)

    def __iter__(self):
        while True:
            item = self.items.get()
            if item is None:
                return
            yield item


class FakeProcess:
    def __init__(self, *, returncode=None, stderr=(), reply=True, restart_required=False):
        self.returncode = returncode
        self.poll_count = 0
        self.stdout = QueuePipe(initial=[None] if returncode is not None else ())
        self.stderr = QueuePipe(initial=[line + "\n" for line in stderr] + [None])
        self.stdin = QueuePipe(on_write=self._write)
        self.started = []
        self.reply = reply
        self.restart_required = restart_required

    def _write(self, line):
        request = json.loads(line)
        self.started.append(request)
        if not self.reply:
            return
        self.stdout.items.put(
            json.dumps(
                {
                    "request_id": request["request_id"],
                    "correlation_id": request["correlation_id"],
                    "restart_required": self.restart_required,
                    "result": {
                        "blocks": [],
                        "confidence": None,
                        "warnings": [],
                        "diagnostics": {"page_number": request["page_number"], "model_revision": "test-model"},
                    },
                }
            )
            + "\n"
        )

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = 1

    def kill(self):
        self.returncode = 1


def tiny_png():
    image = pytest.importorskip("PIL.Image")
    output = io.BytesIO()
    image.new("RGB", (8, 8), "white").save(output, format="PNG")
    return output.getvalue()


def test_provider_reuses_worker_across_pages(monkeypatch):
    workers = []

    def start(*args, **kwargs):
        process = FakeProcess()
        workers.append(process)
        return process

    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", start)
    provider = PaddleOCRProvider(max_pixels=1000)
    first = provider.recognize(
        tiny_png(), page_number=1, languages=("vi",), correlation_id="trace", job_id="job", dpi=160
    )
    second = provider.recognize(
        tiny_png(), page_number=2, languages=("vi",), correlation_id="trace", job_id="job", dpi=160
    )
    provider.close_document()

    assert len(workers) == 1
    assert [item["page_number"] for item in workers[0].started] == [1, 2]
    assert first.diagnostics["model_revision"] == "test-model"
    assert second.diagnostics["page_number"] == 2
    assert all(item["correlation_id"] == "trace" and item["job_id"] == "job" for item in workers[0].started)


def test_native_worker_exit_preserves_returncode_and_stderr(monkeypatch):
    process = FakeProcess(returncode=3221225477, stderr=["paddle native failure"])
    process.stdout.items.put(None)
    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", lambda *a, **k: process)
    provider = PaddleOCRProvider(max_pixels=1000)

    with pytest.raises(OCRWorkerError) as caught:
        provider.recognize(
            tiny_png(), page_number=4, languages=("vi",), correlation_id="trace-4", job_id="job-1", dpi=160
        )

    assert caught.value.diagnostics["failure_category"] == "native_worker_exit"
    assert caught.value.diagnostics["returncode"] == 3221225477
    assert "paddle native failure" in caught.value.diagnostics["stderr_tail"]


def test_worker_start_failure_reports_configuration_error_without_retry(monkeypatch):
    calls = []

    def fail_start(*args, **kwargs):
        calls.append(True)
        raise FileNotFoundError("python worker executable unavailable")

    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", fail_start)
    provider = PaddleOCRProvider(max_pixels=1000)
    with pytest.raises(OCRWorkerError) as caught:
        provider.recognize(tiny_png(), page_number=1, languages=("vi",), correlation_id="trace", dpi=160)
    assert caught.value.diagnostics["failure_category"] == "worker_start_failed"
    assert caught.value.diagnostics["error_message"] == "python worker executable unavailable"
    assert len(calls) == 1


def test_worker_restarts_at_bounded_page_batch(monkeypatch):
    workers = []

    def start(*args, **kwargs):
        process = FakeProcess(restart_required=True)
        workers.append(process)
        return process

    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", start)
    provider = PaddleOCRProvider(max_pixels=1000, max_pages_per_worker=1)
    provider.recognize(tiny_png(), page_number=1, languages=("vi",), correlation_id="trace", dpi=160)
    provider.recognize(tiny_png(), page_number=2, languages=("vi",), correlation_id="trace", dpi=160)
    provider.close_document()
    assert len(workers) == 2


def test_only_timeout_gets_one_bounded_retry(monkeypatch):
    workers = []

    def start(*args, **kwargs):
        process = FakeProcess(reply=len(workers) > 0)
        workers.append(process)
        return process

    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", start)
    provider = PaddleOCRProvider(max_pixels=1000, timeout=0.01)
    provider.recognize(tiny_png(), page_number=1, languages=("vi",), correlation_id="trace", dpi=160)
    provider.close_document()
    assert len(workers) == 2


def test_timeout_retry_budget_is_exactly_one(monkeypatch):
    workers = []

    def start(*args, **kwargs):
        process = FakeProcess(reply=False)
        workers.append(process)
        return process

    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", start)
    provider = PaddleOCRProvider(max_pixels=1000, timeout=0.01)
    with pytest.raises(OCRWorkerError) as caught:
        provider.recognize(tiny_png(), page_number=1, languages=("vi",), correlation_id="trace", dpi=160)
    assert caught.value.diagnostics["failure_category"] == "timeout"
    assert len(workers) == 2


def test_provider_error_is_not_retried(monkeypatch):
    process = FakeProcess()
    process.stdin.on_write = lambda line: process.stdout.items.put(
        json.dumps(
            {
                "request_id": json.loads(line)["request_id"],
                "error": {
                    "category": "provider_error",
                    "type": "RuntimeError",
                    "message": "bad model",
                    "diagnostics": {},
                },
            }
        )
        + "\n"
    )
    starts = []
    monkeypatch.setattr("app.ingestion.ocr.paddle.subprocess.Popen", lambda *a, **k: starts.append(process) or process)
    provider = PaddleOCRProvider(max_pixels=1000)
    with pytest.raises(OCRWorkerError) as caught:
        provider.recognize(tiny_png(), page_number=2, languages=("vi",), correlation_id="trace", dpi=160)
    assert caught.value.diagnostics["failure_category"] == "provider_error"
    assert len(starts) == 1
