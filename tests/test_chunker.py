from app.ingestion.chunker import TextChunker
from app.schemas.document import Document


def test_chunker_preserves_heading_and_required_metadata():
    document = Document(
        doc_id="doc-hypertension",
        title="Tang huyet ap",
        source_path="data/sample/hypertension.md",
        source_type="md",
        text="# Tang huyet ap\n\nTang huyet ap la yeu to nguy co tim mach.\n\n## Dieu tri\n\nCan hoc theo guideline.",
        metadata={"created_at": "2026-05-19T00:00:00Z"},
    )

    chunks = TextChunker(chunk_size=80, chunk_overlap=10).chunk(document)

    assert chunks
    assert chunks[0].metadata.doc_id == "doc-hypertension"
    assert chunks[0].metadata.source_type == "md"
    assert chunks[0].metadata.section_title == "Tang huyet ap"
    assert chunks[0].metadata.hash
