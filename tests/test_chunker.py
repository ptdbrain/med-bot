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


def test_chunker_prefers_sentence_boundaries_and_tracks_positions():
    document = Document(
        doc_id="doc-enzymes",
        title="Enzymes",
        source_path="data/raw/enzymes.md",
        source_type="md",
        text=(
            "# Enzyme kinetics\n\n"
            "Enzymes lower activation energy. "
            "Substrate concentration affects reaction velocity. "
            "Temperature and pH also influence enzyme activity. "
            "Regulation keeps metabolic pathways coordinated."
        ),
        metadata={"created_at": "2026-07-31T00:00:00Z"},
    )

    chunks = TextChunker(
        chunk_size=95,
        chunk_overlap=25,
        min_chunk_size=30,
    ).chunk(document)

    assert len(chunks) >= 2
    assert all(chunk.text.endswith((".", "!", "?")) for chunk in chunks[:-1])
    assert [chunk.metadata.extra["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata.extra["char_start"] < chunk.metadata.extra["char_end"] for chunk in chunks)


def test_chunker_absorbs_small_tail_instead_of_creating_tiny_chunk():
    text = (
        "First sentence contains enough useful context for retrieval. "
        "Second sentence adds supporting evidence. "
        "Short tail."
    )
    document = Document(
        doc_id="doc-tail",
        title="Tail handling",
        source_path="data/raw/tail.txt",
        source_type="txt",
        text=text,
    )

    chunks = TextChunker(
        chunk_size=105,
        chunk_overlap=15,
        min_chunk_size=35,
    ).chunk(document)

    assert chunks[-1].text.endswith("Short tail.")
    assert len(chunks[-1].text) >= 35
