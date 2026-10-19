import re

from tests.rag_helpers import make_document


class Counter:
    name, revision = "words", "1"

    def offsets(self, text):
        return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]

    def count(self, text):
        return len(self.offsets(text))


def test_chunks_fit_budget_and_preserve_parent_and_offsets():
    from app.ingestion.token_chunker import TokenChunker

    document = make_document("Enzymes lower activation energy. " * 30)
    counter = Counter()
    chunks = TokenChunker(counter, target_tokens=20, overlap_tokens=4, min_tokens=5, max_tokens=24).chunk(document)
    assert len(chunks) > 2
    assert all(c.metadata.extra["indexed_token_count"] <= 24 for c in chunks)
    assert all(c.metadata.extra["token_count"] == counter.count(c.text) for c in chunks)
    parent = document.metadata["parents"][chunks[0].metadata.parent_id]
    assert parent["text"] == document.text.strip()
    assert all(parent["text"][c.metadata.extra["char_start"] : c.metadata.extra["char_end"]] == c.text for c in chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in TokenChunker(counter, 20, 4, 5, 24).chunk(document)]


def test_short_definition_is_not_deleted():
    from app.ingestion.token_chunker import TokenChunker

    chunks = TokenChunker(Counter()).chunk(make_document("ATP transfers energy."))
    assert len(chunks) == 1
    assert "short_semantic_unit" in chunks[0].metadata.extra["warnings"]


def test_long_single_sentence_makes_progress_without_dropping_text():
    from app.ingestion.token_chunker import TokenChunker

    text = " ".join("t" + str(i) for i in range(80))
    chunks = TokenChunker(Counter(), 20, 4, 5, 24).chunk(make_document(text))
    assert chunks[-1].text.endswith("t79")
    assert len(chunks) < 10
    assert all(c.metadata.extra["indexed_token_count"] <= 24 for c in chunks)
