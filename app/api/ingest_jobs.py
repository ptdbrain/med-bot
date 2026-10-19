from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.jobs.ingest_jobs import IngestJobQueue
from app.utils.file_utils import ALLOWED_DOCUMENT_EXTENSIONS, validate_extension, validate_size

router = APIRouter(prefix="/ingest/jobs", tags=["ingestion jobs"])


def queue_for(request):
    settings = request.app.state.rag_service.settings
    return IngestJobQueue(settings.metadata_db_path, settings.data_dir / "processed" / "job_uploads")


def public_job(job):
    return {k: job[k] for k in ("job_id", "filename", "status", "report", "error", "cancel_requested")}


@router.post("", status_code=202)
async def enqueue(
    request: Request,
    file: UploadFile = File(...),
    replace_doc_id: str | None = Form(None),
    allow_partial: bool = Form(False),
):
    limit = request.app.state.rag_service.settings.max_upload_bytes
    content = await file.read(limit + 1)
    try:
        validate_size(content, limit)
        validate_extension(file.filename or "", ALLOWED_DOCUMENT_EXTENSIONS)
        job = await run_in_threadpool(
            queue_for(request).enqueue,
            file.filename or "upload.txt",
            content,
            replace_doc_id=replace_doc_id,
            allow_partial=allow_partial,
        )
        return public_job(job)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/{job_id}")
def get_job(job_id: str, request: Request):
    try:
        return public_job(queue_for(request).get(job_id))
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    try:
        queue_for(request).cancel(job_id)
        return get_job(job_id, request)
    except KeyError as exc:
        raise HTTPException(404, "Job not found") from exc


@router.post("/{job_id}/retry")
def retry_job(job_id: str, request: Request, allow_partial: bool = False):
    try:
        return public_job(queue_for(request).retry(job_id, allow_partial=allow_partial))
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
