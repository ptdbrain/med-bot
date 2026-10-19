import pytest


def test_terminal_job_cannot_return_to_embedding():
    from app.jobs.ingest_jobs import require_transition

    with pytest.raises(ValueError):
        require_transition("completed", "embedding")


def test_jobs_survive_restart_and_claim_has_one_owner(tmp_path):
    from app.jobs.ingest_jobs import IngestJobQueue

    first = IngestJobQueue(tmp_path / "jobs.db", tmp_path / "uploads")
    job = first.enqueue("notes.md", b"Document evidence")
    restarted = IngestJobQueue(tmp_path / "jobs.db", tmp_path / "uploads")
    assert restarted.claim("one")["job_id"] == job["job_id"]
    assert first.claim("two") is None
    with pytest.raises(ValueError):
        first.update(job["job_id"], "two", "embedding")


def test_cancelled_job_does_not_run(tmp_path):
    from app.jobs.ingest_jobs import IngestJobQueue

    queue = IngestJobQueue(tmp_path / "jobs.db", tmp_path / "uploads")
    job = queue.enqueue("notes.md", b"Evidence")
    queue.cancel(job["job_id"])
    assert queue.claim("one") is None
    assert queue.get(job["job_id"])["status"] == "cancelled"


def test_run_one_propagates_job_id_as_ocr_trace_context(tmp_path):
    from dataclasses import dataclass

    from app.jobs.ingest_jobs import IngestJobQueue

    @dataclass
    class Result:
        doc_id: str

    class Service:
        received = None

        def ingest_file(self, filename, content, *, replace_doc_id, allow_partial, progress, correlation_id, job_id):
            self.received = (correlation_id, job_id)
            progress("chunking", {})
            progress("embedding", {})
            progress("publishing", {})
            return Result("doc-1")

    queue = IngestJobQueue(tmp_path / "jobs.db", tmp_path / "uploads")
    queued = queue.enqueue("scan.pdf", b"pdf")
    service = Service()
    result = queue.run_one(service)
    assert service.received == (queued["job_id"], queued["job_id"])
    assert result["status"] == "completed"


def test_failed_pdf_job_keeps_extraction_report_and_error(tmp_path):
    from app.ingestion.pdf_loader import PDFExtractionError
    from app.jobs.ingest_jobs import IngestJobQueue

    class Service:
        def ingest_file(self, filename, content, **kwargs):
            raise PDFExtractionError(
                "PDF contains no usable text (page 1: ocr_failed)",
                {"correlation_id": kwargs["correlation_id"], "job_id": kwargs["job_id"], "pages": []},
            )

    queue = IngestJobQueue(tmp_path / "jobs.db", tmp_path / "uploads")
    queued = queue.enqueue("scan.pdf", b"pdf")
    result = queue.run_one(Service())
    assert result["status"] == "failed"
    assert result["error"].endswith("page 1: ocr_failed)")
    assert result["report"]["job_id"] == queued["job_id"]
