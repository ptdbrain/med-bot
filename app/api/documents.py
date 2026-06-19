from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request


router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def list_documents(request: Request) -> dict:
    documents = request.app.state.rag_service.list_documents()
    return {"documents": documents, "total": len(documents)}


@router.delete("/{doc_id}")
def delete_document(doc_id: str, request: Request) -> dict:
    deleted = request.app.state.rag_service.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu.")
    return {"doc_id": doc_id, "deleted": True}
