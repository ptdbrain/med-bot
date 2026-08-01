from app.retrieval.relevance import filter_relevant_results
from app.schemas.document import Chunk, ChunkMetadata, SearchResult


def _result(chunk_id: str, title: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=Chunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                doc_id=chunk_id,
                source_path=f"data/raw/{chunk_id}.md",
                source_type="md",
                title=title,
                created_at="2026-06-13T00:00:00Z",
                hash=chunk_id,
            ),
        ),
        score=0.9,
    )


def test_relevance_ignores_question_scaffolding_terms() -> None:
    hypertension = _result(
        "hypertension",
        "Tăng huyết áp",
        "Thay đổi lối sống có vai trò quan trọng.",
    )
    blood = _result(
        "blood",
        "Xét nghiệm máu",
        "Hemoglobin phản ánh khả năng vận chuyển oxy.",
    )

    filtered = filter_relevant_results(
        "Hemoglobin có vai trò gì?",
        [hypertension, blood],
    )

    assert [result.chunk.chunk_id for result in filtered] == ["blood"]


def test_relevance_rejects_generic_overlap_without_topic_anchor() -> None:
    generic = _result(
        "generic",
        "Giáo dục sức khỏe",
        "Hoạt động thể chất phù hợp hỗ trợ giáo dục y khoa.",
    )

    filtered = filter_relevant_results(
        "Insulin hoạt động thế nào?",
        [generic],
    )

    assert filtered == []


def test_relevance_accepts_cross_language_semantic_match() -> None:
    english_context = _result(
        "enzyme",
        "Medical Biochemistry",
        "Enzymes lower activation energy in metabolic reactions.",
    )
    english_context.signals["semantic_score"] = 0.82

    filtered = filter_relevant_results(
        "Enzyme có vai trò gì trong chuyển hóa?",
        [english_context],
    )

    assert filtered == [english_context]


def test_relevance_rejects_weak_semantic_match_without_lexical_anchor() -> None:
    unrelated = _result(
        "unrelated",
        "Anatomy",
        "The skeletal system provides structural support.",
    )
    unrelated.signals["semantic_score"] = 0.20

    filtered = filter_relevant_results(
        "Enzyme có vai trò gì trong chuyển hóa?",
        [unrelated],
    )

    assert filtered == []
