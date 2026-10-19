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
