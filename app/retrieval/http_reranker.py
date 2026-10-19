import math

from app.providers.http_transport import HTTPTransport
from app.retrieval.reranker import normalize_logit
from app.schemas.document import SearchResult


class HTTPReranker:
    def __init__(self, settings, transport=None):
        self.transport = transport or HTTPTransport(settings)

    def rerank(self, query, results, top_k=5):
        if not results or top_k <= 0:
            return []
        response = self.transport.post(
            "/v1/rerank",
            {"query": query, "documents": [{"id": r.chunk.chunk_id, "text": r.chunk.text} for r in results]},
        )
        by_id = {r.chunk.chunk_id: r for r in results}
        rows = response.get("results", [])
        if {row["id"] for row in rows} != set(by_id) or len(rows) != len(by_id):
            raise ValueError("Reranker result IDs do not match candidates")
        output = []
        for row in rows:
            raw = float(row["score"])
            kind = response.get("score_type")
            if kind == "logit":
                score = normalize_logit(raw)
            elif kind == "normalized" and math.isfinite(raw) and 0 <= raw <= 1:
                score = raw
            else:
                raise ValueError("Invalid remote score type or value")
            old = by_id[row["id"]]
            output.append(SearchResult(old.chunk, score, {**old.signals, "reranker_score": score}))
        return sorted(output, key=lambda item: (-item.score, item.chunk.chunk_id))[:top_k]
