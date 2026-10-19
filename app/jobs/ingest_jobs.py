from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

TERMINAL = {"completed", "needs_review", "failed", "cancelled"}
STAGES = ["queued", "extracting", "chunking", "embedding", "publishing", "completed"]


def require_transition(old: str, new: str) -> None:
    if old in TERMINAL or new not in set(STAGES) | TERMINAL:
        raise ValueError("Invalid job transition")
    if new not in TERMINAL and STAGES.index(new) < STAGES.index(old):
        raise ValueError("Job stage cannot move backwards")


class IngestJobQueue:
    def __init__(self, path: Path, uploads: Path):
        self.path, self.uploads = Path(path), Path(uploads)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.uploads.mkdir(parents=True, exist_ok=True)
        with self._connect() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS ingest_jobs(job_id TEXT PRIMARY KEY,filename TEXT,path TEXT,status TEXT,owner TEXT,lease REAL,replace_doc_id TEXT,allow_partial INTEGER,cancel_requested INTEGER,report TEXT,error TEXT)"
            )

    @contextmanager
    def _connect(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    def enqueue(self, filename: str, content: bytes, *, replace_doc_id=None, allow_partial=False):
        identifier = str(uuid.uuid4())
        filename = Path(filename.replace("\\", "/")).name
        path = self.uploads / (identifier + Path(filename).suffix.lower())
        path.write_bytes(content)
        with self._connect() as c:
            c.execute(
                "INSERT INTO ingest_jobs VALUES(?,?,?,'queued',NULL,0,?,?,0,'{}',NULL)",
                (identifier, filename, str(path), replace_doc_id, int(allow_partial)),
            )
        return self.get(identifier)

    def get(self, identifier):
        with self._connect() as c:
            row = c.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (identifier,)).fetchone()
        if not row:
            raise KeyError(identifier)
        result = dict(row)
        result["report"] = json.loads(result["report"])
        return result

    def claim(self, owner, lease_seconds=120):
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT job_id FROM ingest_jobs WHERE cancel_requested=0 AND (status='queued' OR (status IN ('extracting','chunking','embedding','publishing') AND lease<?)) ORDER BY rowid LIMIT 1",
                (time.time(),),
            ).fetchone()
            if not row:
                return None
            c.execute(
                "UPDATE ingest_jobs SET status='extracting',owner=?,lease=? WHERE job_id=?",
                (owner, time.time() + lease_seconds, row[0]),
            )
        return self.get(row[0])

    def update(self, identifier, owner, status, *, report=None, error=None):
        with self._connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM ingest_jobs WHERE job_id=?", (identifier,)).fetchone()
            if row is None or row["owner"] != owner or row["lease"] < time.time():
                raise ValueError("Job lease is not owned")
            if row["cancel_requested"] and status not in TERMINAL:
                raise InterruptedError("Ingest cancelled")
            require_transition(row["status"], status)
            c.execute(
                "UPDATE ingest_jobs SET status=?,report=?,error=?,lease=? WHERE job_id=?",
                (status, json.dumps(report or {}, ensure_ascii=False), error, time.time() + 120, identifier),
            )

    def heartbeat(self, identifier, owner):
        with self._connect() as c:
            c.execute(
                "UPDATE ingest_jobs SET lease=? WHERE job_id=? AND owner=? AND lease>=?",
                (time.time() + 120, identifier, owner, time.time()),
            )

    def cancel(self, identifier):
        job = self.get(identifier)
        if job["status"] in TERMINAL:
            return
        with self._connect() as c:
            c.execute(
                "UPDATE ingest_jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE job_id=?",
                (identifier,),
            )

    def retry(self, identifier, *, allow_partial=False):
        with self._connect() as c:
            changed = c.execute(
                "UPDATE ingest_jobs SET status='queued',owner=NULL,lease=0,cancel_requested=0,error=NULL,allow_partial=? WHERE job_id=? AND status IN ('failed','needs_review','cancelled')",
                (int(allow_partial), identifier),
            ).rowcount
        if not changed:
            raise ValueError("Only failed/review/cancelled jobs can be retried")
        return self.get(identifier)

    def run_one(self, service):
        owner = str(uuid.uuid4())
        job = self.claim(owner)
        if job is None:
            return None
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(20):
                self.heartbeat(job["job_id"], owner)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:

            def progress(stage, report):
                self.update(job["job_id"], owner, stage, report=report)

            result = service.ingest_file(
                job["filename"],
                Path(job["path"]).read_bytes(),
                replace_doc_id=job["replace_doc_id"],
                allow_partial=bool(job["allow_partial"]),
                progress=progress,
            )
            self.update(job["job_id"], owner, "completed", report=asdict(result))
        except InterruptedError:
            self.update(job["job_id"], owner, "cancelled")
        except Exception as exc:  # noqa: BLE001 - Persist all worker failures for inspection and retry.
            review = getattr(exc, "report", None)
            self.update(
                job["job_id"], owner, "needs_review" if review else "failed", report=review, error=type(exc).__name__
            )
        finally:
            stopped.set()
            thread.join(timeout=2)
        return self.get(job["job_id"])
