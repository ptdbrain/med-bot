from __future__ import annotations

import urllib.request

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, HttpUrl

from app.ingestion.html_loader import HtmlLoader
from app.utils.hashing import short_hash

router = APIRouter(prefix="/ingest", tags=["ingest"])


class UrlIngestRequest(BaseModel):
    url: HttpUrl


@router.post("/file")
async def ingest_file(request: Request, file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        result = request.app.state.rag_service.ingest_file(file.filename or "upload.txt", content)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.__dict__


@router.post("/url")
def ingest_url(payload: UrlIngestRequest, request: Request) -> dict:
    try:
        with urllib.request.urlopen(str(payload.url), timeout=15) as response:
            content = response.read()
    except Exception as exc:  # pragma: no cover - network-dependent
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc

    filename = f"url-{short_hash(str(payload.url))}.html"
    document = HtmlLoader().load(request.app.state.rag_service.settings.data_dir / "raw" / filename, content)
    document.metadata["url"] = str(payload.url)
    result = request.app.state.rag_service.pipeline.ingest_document(document)
    return result.__dict__
