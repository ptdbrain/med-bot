from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.ingestion.pipeline import IngestResult
from app.jobs.crawler import PublicHtmlCrawler
from app.utils.hashing import stable_hash

if TYPE_CHECKING:
    from app.rag.service import RAGService


def has_changed(previous_hash: str | None, content: bytes) -> tuple[bool, str]:
    current_hash = stable_hash(content.decode("utf-8", errors="replace"))
    return previous_hash != current_hash, current_hash


@dataclass(frozen=True)
class UrlUpdateResult:
    url: str
    changed: bool
    content_hash: str
    doc_id: str
    title: str
    chunks_added: int
    source_path: str
    source_type: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "changed": self.changed,
            "content_hash": self.content_hash,
            "doc_id": self.doc_id,
            "title": self.title,
            "chunks_added": self.chunks_added,
            "source_path": self.source_path,
            "source_type": self.source_type,
        }


class IncrementalUrlUpdater:
    def __init__(
        self,
        service: RAGService,
        crawler=None,
    ) -> None:
        self.service = service
        self.crawler = crawler or PublicHtmlCrawler()

    def update(self, url: str) -> UrlUpdateResult:
        return self.update_content(url, self.crawler.fetch(url))

    def update_content(self, url: str, content: bytes) -> UrlUpdateResult:
        state = self.service.metadata_store.get_url_source(url)
        changed, content_hash = has_changed(
            state["content_hash"] if state else None,
            content,
        )
        if not changed and state:
            document = self.service.metadata_store.get_document(state["doc_id"])
            if document is not None:
                return self._result_from_document(
                    url,
                    False,
                    content_hash,
                    document,
                )

        ingest_result = self.service.ingest_url(url, content)
        previous_doc_id = state["doc_id"] if state else None
        if previous_doc_id and previous_doc_id != ingest_result.doc_id:
            self.service.delete_document(previous_doc_id)
        self.service.metadata_store.upsert_url_source(
            url,
            content_hash,
            ingest_result.doc_id,
        )
        return self._result_from_ingest(
            url,
            True,
            content_hash,
            ingest_result,
        )

    def _result_from_ingest(
        self,
        url: str,
        changed: bool,
        content_hash: str,
        result: IngestResult,
    ) -> UrlUpdateResult:
        return UrlUpdateResult(
            url=url,
            changed=changed,
            content_hash=content_hash,
            doc_id=result.doc_id,
            title=result.title,
            chunks_added=result.chunks_added,
            source_path=result.source_path,
            source_type=result.source_type,
        )

    def _result_from_document(
        self,
        url: str,
        changed: bool,
        content_hash: str,
        document: dict,
    ) -> UrlUpdateResult:
        return UrlUpdateResult(
            url=url,
            changed=changed,
            content_hash=content_hash,
            doc_id=document["doc_id"],
            title=document["title"],
            chunks_added=document["chunks_count"],
            source_path=document["source_path"],
            source_type=document["source_type"],
        )
