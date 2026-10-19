from __future__ import annotations

import urllib.request

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, HttpUrl

from app.jobs.crawler import validate_public_url
from app.jobs.incremental_update import IncrementalUrlUpdater

router = APIRouter(prefix="/ingest", tags=["ingest"])


class UrlIngestRequest(BaseModel):
    url: HttpUrl


@router.post("/file")
async def ingest_file(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read(request.app.state.rag_service.settings.max_upload_bytes + 1)
    try:
        result = await run_in_threadpool(
            request.app.state.rag_service.ingest_file, file.filename or "upload.txt", content
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.__dict__


@router.post("/url")
def ingest_url(payload: UrlIngestRequest, request: Request) -> dict:
    url = str(payload.url)
    try:
        validate_public_url(url)
        with urllib.request.urlopen(url, timeout=15) as response:
            content = response.read()
    except Exception as exc:  # pragma: no cover - network-dependent
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc

    try:
        result = IncrementalUrlUpdater(
            request.app.state.rag_service,
        ).update_content(
            url,
            content,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.to_dict()
