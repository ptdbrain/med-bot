# RAG Quality Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:subagent-driven-development only when delegation is requested. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hoàn thiện ingest PDF text/scan, chunking, retrieval Việt–Anh và câu trả lời có căn cứ, với benchmark tái lập và khả năng đổi local/cloud provider.

**Architecture:** Giữ kiến trúc modular hiện tại; bổ sung page/block extraction, token chunker, index generation và answer validation qua interface. Một pipeline dùng các profile local/cloud/hybrid, nhận cùng hợp đồng dữ liệu. Đánh giá theo từng tầng để biết lỗi xuất phát từ OCR, chunking, retrieval hay generation.

**Tech Stack:** Python >=3.11, FastAPI, SQLite metadata, Qdrant hoặc SQLite dense, pypdf, PDFium renderer, PaddleOCR multilingual, Sentence Transformers/BGE-M3, BM25, multilingual cross-encoder, Ollama, Next.js/TypeScript, pytest/Vitest.

**Approved Design:** [2026-09-19-rag-quality-design.md](../specs/2026-09-19-rag-quality-design.md), đã được người dùng duyệt trong hội thoại; commit thiết kế `43f1371`.

**Status:** Chỉ lập kế hoạch. Các checkbox dưới đây chưa thể hiện chức năng đã được triển khai. Không dùng các chỉ số lịch sử trong README làm kết quả nghiệm thu cho phiên bản này.

## Global Constraints

- Keep the modular architecture. Do not collapse ingestion, retrieval, safety, LLM, API, and UI logic into one file.
- Run relevant tests for every change. When adding a RAG mode, add or update tests and README sections.
- Provider integrations must go through an interface or adapter. Do not hard-code OpenAI, Qdrant, OCR, STT, or vision-provider calls directly inside API routes.
- Do not commit `.env`, API keys, real patient records, or sensitive medical data.
- The product is a medical education assistant, not a diagnostic or prescribing system.
- Every medical answer must preserve the disclaimer: `Thông tin chỉ phục vụ học tập/tham khảo, không thay thế tư vấn của bác sĩ.`
- Emergency symptoms must trigger an emergency/care-seeking response before retrieval.
- If retrieval has no relevant context, answer that the provided documents do not contain enough information. Do not invent citations.
- Test mặc định phải chạy offline; tải model, API có phí và test với dịch vụ thật phải được bật riêng.
- Không sửa/xóa index hiện có để thử thuật toán. Dùng namespace, collection và thư mục benchmark riêng; giữ đường rollback.
- Mỗi task: regression test có ý nghĩa -> xác nhận FAIL -> implementation nhỏ nhất -> xác nhận PASS -> review diff. Commit theo task khi được yêu cầu; chỉ stage các file thuộc task.

---

## 1. Kết Quả Cần Bàn Giao

| Milestone | Task | Sản phẩm kiểm tra được |
| --- | --- | --- |
| M0: Có cách chạy và đo | 1–2 | Runtime doctor; corpus/nhãn có schema, split và provenance |
| M1: Đọc và chia tài liệu tốt | 3–6 | PDF text/scan/mixed -> blocks -> chunks kèm báo cáo chất lượng |
| M2: Ingest an toàn | 7–8 | Publish phiên bản mới, resume, rollback, trạng thái trên UI |
| M3: Truy hồi đúng | 9–11 | Hybrid + rerank + relevance + context budget, giữ được nguồn |
| M4: Trả lời có căn cứ | 12–13 | Citation hợp lệ, kiểm tra claim, repair tối đa một lần |
| M5: Đổi provider và nghiệm thu | 14–16 | Cloud adapters, benchmark, báo cáo và runbook có bằng chứng |

Thứ tự chính: `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 15 -> 16`.
Task 2 được tạo trước, nhưng nhãn page/evidence chỉ đóng băng sau khi tài liệu được kiểm tra ở Task 7. Có thể làm contract test cloud khi provider local đang chạy benchmark; không chỉnh chung file giữa hai người thực hiện.

Mốc hoàn thành local là M4 cùng benchmark local của Task 15. Task 15 chạy được sau Task 13, không phải chờ tài khoản cloud ở Task 14. M5 có thêm nghiệm thu cloud riêng; test mock HTTP không được báo thành live cloud PASS.

## 2. Những Điều Cần Sửa Từ Baseline

Các quan sát dưới đây dựa trên source, không phải kết quả chạy thử mới:

| File hiện tại | Khoảng trống | Task |
| --- | --- | --- |
| `app/ingestion/pdf_loader.py` | Chỉ extract text, chưa có OCR/page report | 3–5 |
| `app/ingestion/base.py` | `doc_id` hash cả path và text, sửa nội dung tạo ID khác | 3, 7 |
| `app/ingestion/chunker.py` | Window theo ký tự; chưa có token budget/block contract | 6 |
| `app/ingestion/pipeline.py` | Ghi raw trước validation; ghi vector và metadata không chung transaction | 7 |
| `app/retrieval/metadata_store.py` | Parent ghép các chunk overlap; có thể lặp nội dung | 6–7, 11 |
| `app/retrieval/bm25_store.py` | Tính lại thống kê toàn corpus mỗi query | 9 |
| `app/rag/hybrid_rag.py`, `rerank_rag.py` | Candidate multiplier lồng nhau, khó khống chế chi phí | 9–10 |
| `app/retrieval/reranker.py` | Chưa ép raw logits trước sigmoid; ghi đè semantic signal | 10 |
| `app/retrieval/relevance.py` | Threshold cố định, một token trùng cũng có thể qua gate | 10 |
| `app/retrieval/query_transform.py` | Thêm tiền tố chung chung, chưa decomposition theo ý | 11 |
| `app/evaluation/retrieval_eval.py` | Match substring tên tài liệu; chưa xác nhận đoạn thực sự có đáp án | 2, 15 |
| `app/evaluation/answer_eval.py` | Có `[n]` chưa có nghĩa claim được hỗ trợ; bốn số chưa chứng minh bốn phần đúng | 12–15 |
| `app/llm/openai_llm.py` | Boundary còn `NotImplementedError`, chưa phải cloud integration | 14 |

Không tự động khôi phục Ollama hoặc tải thêm model trong bước viết plan. Task 1 phải kiểm tra lại runtime, ổ đĩa và model trước khi triển khai.

## 3. Quyết Định Kỹ Thuật Và Giới Hạn

1. **Token thay ký tự:** target 500, overlap mục tiêu 80, minimum advisory 100, hard maximum 640 token. Đếm cả title/heading được thêm vào embedding. Thực nghiệm 400/500/600 ở Task 15; không mặc định một giá trị là tối ưu.
2. **Không làm mất định nghĩa ngắn:** chunk dưới minimum được ghép trong cùng section nếu phù hợp; định nghĩa/bảng nhỏ có ý nghĩa được giữ và gắn `short_semantic_unit`. Rỗng/rác bị loại. Minimum không phải ngưỡng xóa mọi đoạn ngắn.
3. **Page mixed:** nhiều ký tự không chứng minh text layer đầy đủ. Kiểm tra vùng ảnh lớn và phần text thiếu; OCR vùng nghi ngờ, dedup theo vị trí. Không bỏ hình/bảng có chữ chỉ vì trang có một đoạn text tốt.
4. **Dấu tiếng Việt:** dense query giữ Unicode có dấu; sparse có nhánh accent-folded. Không tự đoán dấu rồi thay câu hỏi gốc. Bản rewrite phải bảo toàn thuật ngữ, phủ định, số và đơn vị.
5. **Index compatibility:** đổi OCR/parser/chunker có thể cần extract/rechunk; đổi embedding model/revision/pooling/normalization/prefix/dimension bắt buộc re-embed. Đổi LLM/reranker không nhất thiết rebuild vectors, nhưng phải benchmark lại.
6. **Không có transaction chung SQLite–Qdrant:** dùng dữ liệu phiên bản bất biến, trạng thái staging và publication pointer; query chỉ thấy phiên bản đã publish. Atomic ở lớp hiển thị, không tuyên bố transaction ACID xuyên hai hệ thống.
7. **Faithfulness:** validator tự động là bộ lọc có sai số. Chỉ số chấp nhận cần đối chiếu nhãn claim/evidence do người đọc kiểm tra; không dùng chính LLM sinh đáp án làm nguồn ground truth duy nhất.
8. **Cloud:** profile chỉ cấu hình provider, không tự gửi tài liệu ra ngoài khi local lỗi. Live cloud dùng corpus test không nhạy cảm, endpoint/model cụ thể và giới hạn request/token.
9. **Layout:** plain OCR trả dòng chữ chưa đủ để khôi phục bảng. Dùng bounding boxes và layout/table adapter khi cần; bảng không khôi phục chắc chắn phải báo `table_structure_uncertain`.
10. **Chưa có tài liệu riêng:** bắt đầu bằng tài liệu test tự soạn Việt–Anh, xuất cả text PDF và scan cùng nội dung. Kết quả trên dữ liệu tự soạn chỉ là baseline; khi có corpus mục tiêu phải đo lại.

## 4. Bản Đồ File Và Hợp Đồng

Giữ `Document`, `Chunk`, `SearchResult` làm boundary công khai, bổ sung field có default để đọc được dữ liệu cũ. Những chữ ký sau là **API dự kiến được tạo trong plan**, chưa có trong repo.

| File mới/được sửa | Trách nhiệm |
| --- | --- |
| `app/config.py`, `app/runtime/doctor.py` | Profile, validation config, readiness |
| `app/schemas/extraction.py` | Page, block, span, confidence, ingest report |
| `app/ingestion/identity.py` | Logical document ID và revision ID |
| `app/ingestion/ocr/base.py`, `paddle.py`, `factory.py` | OCR interface và local adapter |
| `app/ingestion/pdf_renderer.py`, `pdf_loader.py` | Render/page routing |
| `app/ingestion/normalizer.py`, `layout.py` | Text cleanup, reading order và structured blocks |
| `app/ingestion/tokenizer.py`, `token_chunker.py`, `chunk_quality.py` | Token offsets, structure-aware split, quality report |
| `app/retrieval/index_manifest.py`, `publication.py` | Index compatibility, staging/publish/rollback |
| `app/retrieval/metadata_store.py`, `dense_store.py`, `vector_store.py` | Persistent revisions và active-only search |
| `app/jobs/ingest_jobs.py` | Durable job, lease, progress và resume |
| `app/retrieval/query_normalizer.py`, `bm25_store.py` | Query variants, sparse statistics |
| `app/retrieval/relevance.py`, `reranker.py` | Score semantics, configurable acceptance |
| `app/rag/context_builder.py` | Context budget và source IDs |
| `app/rag/answer_service.py`, `answer_validation.py`, `support_verifier.py` | Shared generation, validation, repair |
| `app/providers/http_transport.py` và adapter cloud mỗi tầng | Hosted inference qua contract rõ ràng |
| `app/evaluation/dataset.py`, `metrics.py`, `benchmark.py` | Labels, metrics, ablation, reports |
| `scripts/doctor.py`, `ingest_directory.py`, `run_benchmark.py`, `migrate_index.py`, `smoke_rag.py` | CLI vận hành |
| `web/app/ingest/page.tsx`, `documents/page.tsx`, `evaluation/page.tsx` | Trạng thái ingest, bằng chứng nguồn, kết quả benchmark |

Hợp đồng dữ liệu chính, chi tiết thêm trong task tương ứng:

```python
from dataclasses import dataclass
from typing import Literal, Protocol

@dataclass(frozen=True)
class SourceSpan:
    page_number: int                 # 1-based PDF page; không phải số trang in
    block_id: str
    start: int                       # offset trong normalized block text
    end: int                         # exclusive

@dataclass(frozen=True)
class TextBlock:
    block_id: str
    page_number: int
    kind: Literal["heading", "paragraph", "list", "table", "caption"]
    text: str
    bbox: tuple[float, float, float, float] | None = None
    method: Literal["text", "ocr", "mixed"] = "text"
    confidence: float | None = None  # None khác 0.0

@dataclass(frozen=True)
class OCRResult:
    blocks: tuple[TextBlock, ...]
    confidence: float | None
    warnings: tuple[str, ...] = ()

class OCRProvider(Protocol):
    name: str
    def recognize(self, image: bytes, *, page_number: int,
                  languages: tuple[str, ...]) -> OCRResult: ...

class TokenCounter(Protocol):
    name: str
    revision: str
    def count(self, text: str) -> int: ...
    def offsets(self, text: str) -> list[tuple[int, int]]: ...
```

`PageExtraction` bổ sung `page_number`, `classification` (`text|scan|mixed|blank`), `blocks`, `warnings`, `metrics: dict[str, float]`, `status` (`usable|review|failed|blank`).
`ExtractionReport` chứa danh sách page reports, tổng pages và digest config; `IngestReport` thêm `doc_id`, `revision_id`, status, chunk stats, durations và publication ID.

`ChunkMetadata.extra` giữ backward-compatible fields: `source_spans`, `token_count`, `token_start/end`, `offset_space`, `language`, `extraction_method`, `ocr_confidence`, `document_hash`, `chunker_version`, `embedding_version`, `revision_id`. Sau migration mới cân nhắc field top-level.

`SearchResult.signals` tiếp tục chỉ chứa số; lý do loại/nhận nằm trong `diagnostics` riêng, không nhét string vào `dict[str, float]`.

## 5. Quy Trình Test Chung

Đứng tại `D:\Project\RAG`. Các lệnh dùng interpreter trong virtualenv của dự án sau Task 1; không hard-code Miniconda vào code/runbook dùng chung.

```powershell
python -m pytest -q
python -m ruff check app scripts tests
python -m black --check app scripts tests
```

Task có UI chạy từ `web`:

```powershell
npm test
npm run lint
npm run build
```

Test offline dùng dependency injection và thư mục tạm. Marker `ocr_live`, `models_live`, `qdrant_live`, `cloud_live` được khai báo trong `pyproject.toml`; mỗi marker chỉ chạy khi có cờ bật và prerequisite tương ứng. Không âm thầm thay model thật bằng mock trong báo cáo.

Tất cả tên file test và CLI mới trong phần task phải được tạo trước khi chạy; failure ban đầu phải do thiếu behavior/module dự kiến, không do sai tên fixture hoặc syntax.

---

## Task 1: Runtime Có Thể Chạy Lại Và Profile Rõ Ràng

**Depends on:** Không có.

**Files:** Modify `app/config.py`, `app/api/health.py`, `.env.example`, `pyproject.toml`, `README.md`; Create `app/runtime/__init__.py`, `app/runtime/doctor.py`, `scripts/doctor.py`, `config/profiles/local.env.example`, `config/profiles/cloud.env.example`, `config/profiles/hybrid.env.example`; Test `tests/test_runtime_doctor.py`, `tests/test_config.py`.

**Interfaces:** `resolve_profile(name: str, overrides: dict[str, str]) -> Settings`; `check_runtime(settings: Settings, *, probe_models: bool = False) -> list[RuntimeCheck]`; `RuntimeCheck(name: str, status: str, message: str)` với status `ok|missing|error|not_checked`.

- [ ] Viết test profile explicit override thắng profile mặc định; local thiếu Ollama không được báo ready; doctor không tự tải model.

```python
def test_profile_keeps_explicit_vector_store():
    from app.config import resolve_profile
    settings = resolve_profile("local", {"VECTOR_STORE_PROVIDER": "sqlite-dense"})
    assert settings.vector_store_provider == "sqlite-dense"
```

- [ ] Chạy `python -m pytest tests/test_runtime_doctor.py tests/test_config.py -q`; ghi FAIL đúng behavior thiếu.
- [ ] Thêm profile lựa chọn provider, kiểm tra lazy import và readiness từng dịch vụ. `demo` giữ mock/lexical để test; `local` dùng PaddleOCR/BGE-M3/cross-encoder/Ollama, default Qdrant; override `sqlite-dense` cho máy chưa chạy Qdrant. `cloud/hybrid` thiếu endpoint hoặc key phải báo lỗi cấu hình, không fallback âm thầm.

```python
def merge_profile(defaults: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    return {**defaults, **{key: value for key, value in overrides.items() if value != ""}}
```

- [ ] Tạo env Windows/Linux bằng Python được thư viện hỗ trợ; kiểm tra wheel OCR, torch, tokenizer trước khi pin dependency đã chạy thật. Lock theo platform, không đoán bản mới nhất. `doctor --json` trả exit 0 khi ready, 1 khi blocker; `--profile` không ghi đè `.env`.
- [ ] Kiểm thử không secrets trong output, host Ollama unreachable, model thiếu, Qdrant unavailable, embedding dimension lệch. Health tách process liveness `/health` và readiness `/ready` trả 503 khi cấu hình bắt buộc thiếu.
- [ ] PASS test và tạo runbook cách chạy mock/local; giữ endpoint localhost mặc định. Chốt milestone bằng `python -m scripts.doctor --profile local --json` có trạng thái thật, kể cả `missing`.

**Gate:** Cài lại theo README xác định được thiếu thư viện/model nào; server không báo ready khi generator bắt buộc không hoạt động.

## Task 2: Corpus Kiểm Thử Và Nhãn Trước Khi Tuning

**Depends on:** 1.

**Files:** Create `app/evaluation/dataset.py`, `evaluation/corpus/manifest.json`, `evaluation/cases/dev.jsonl`, `evaluation/cases/test.jsonl`, `evaluation/README.md`, `scripts/validate_eval_dataset.py`, `scripts/build_rag_fixtures.py`, `tests/fixtures/rag/README.md`, `tests/fixtures/rag/source/`; Test `tests/test_eval_dataset.py`.

**Interfaces:** `load_cases(path: Path) -> list[EvalCase]`; `validate_cases(cases: list[EvalCase], manifest: dict) -> list[str]`. Case fields: `case_id`, `group_id`, `split`, `question`, `query_language`, `category`, `expected`, `relevant_evidence_ids`, `required_claims`, `unsupported_claims`, `reviewed_by`. Manifest map evidence ID -> source digest, PDF page, quote, bbox/span, rights và language.

- [ ] Test cấm cùng `group_id` xuất hiện ở dev và test; cấm answerable không có evidence; không dùng substring tên file làm gold label.

```python
def test_rejects_answerable_without_evidence():
    from app.evaluation.dataset import EvalCase, validate_cases
    case = EvalCase(case_id="bad", group_id="g1", split="dev", question="Q?",
                    query_language="en", category="exact", expected="answerable",
                    relevant_evidence_ids=[], required_claims=[], unsupported_claims=[],
                    reviewed_by="reviewer")
    assert "bad:missing_evidence" in validate_cases([case], {})
```

- [ ] Chạy `python -m pytest tests/test_eval_dataset.py -q`, xác nhận FAIL; triển khai schema và validator trả lỗi có case ID.
- [ ] Tạo 12 tài liệu ngắn tự soạn có quyền sử dụng: 6 Việt, 6 Anh. Xuất text PDF và image-only PDF cùng nội dung; thêm mixed pages, 2 cột, bảng, bullet, xoay 90 độ, scan mờ, trang trắng, PDF mã hóa và PDF hỏng. Lưu nội dung gold và script tạo fixture, không commit giáo trình có quyền sử dụng chưa rõ.
- [ ] Tạo **120 cases** theo phân bố spec: exact 30, paraphrase 30, cross-language 20, multi-hop 15, unanswerable 15, safety 10.
- [ ] Chia theo evidence/topic group trước khi sinh paraphrase: dev 79, test 41; giữ cùng group một phía. Phân bố nghiệm thu:

| Nhóm | Dev | Test |
| --- | ---: | ---: |
| Exact | 20 | 10 |
| Paraphrase | 20 | 10 |
| Cross-language | 14 | 6 |
| Multi-hop | 10 | 5 |
| Unanswerable | 10 | 5 |
| Safety | 5 | 5 |
| **Tổng** | **79** | **41** |

- [ ] Review từng nhãn đối chiếu tài liệu; labels chưa review không tính vào nghiệm thu. OCR comparison dùng bản scan và text tương ứng cùng split. Có ít nhất 4 ca Việt không dấu ở mỗi split. Bộ nhỏ đạt gate chưa đủ chứng minh accuracy 95% ngoài thực tế; báo numerator/denominator.

**Gate:** Validator không lỗi, không leakage, đủ 120 nhãn được review. Nếu chỉ đủ dữ liệu tự soạn, báo rõ corpus synthetic và để gate corpus thực tế mở cho đến khi có tài liệu phù hợp.

## Task 3: Dữ Liệu Page/Block Và Định Danh Ổn Định

**Depends on:** 1–2.

**Files:** Create `app/schemas/extraction.py`, `app/ingestion/identity.py`, `tests/rag_helpers.py`, `tests/__init__.py` nếu chưa có để import helpers không xung đột package `tests` của dependency; Modify `app/schemas/document.py`, `app/ingestion/base.py`; Test `tests/test_extraction_schema.py`, `tests/test_document_identity.py`.

**Interfaces:** Các types ở mục 4. `document_id(collection_id: str, source_key: str) -> str`; `revision_id(content: bytes, extraction_fingerprint: str) -> str`. `Document.metadata["extraction"]` chứa structured report; field cũ vẫn dùng được.

- [ ] Viết test thay nội dung giữ logical document ID, thay source/collection tạo ID khác; schema legacy không có field mới vẫn deserialize.

```python
def test_revision_changes_without_changing_document_identity():
    from app.ingestion.identity import document_id, revision_id
    assert document_id("study", "notes/book.pdf") == document_id("study", "notes/book.pdf")
    assert document_id("study", "notes/book.pdf") != document_id("other", "notes/book.pdf")
    assert revision_id(b"old", "extract-v1") != revision_id(b"new", "extract-v1")
```

- [ ] Chạy `python -m pytest tests/test_extraction_schema.py tests/test_document_identity.py -q`.
- [ ] Implement ID từ canonical source key/collection, tách raw content SHA-256 và extraction fingerprint. Upload mới có document handle; thay tài liệu gửi `replace_doc_id` rõ ràng, không ghi đè chỉ vì cùng filename. URL giữ source key canonical, không lẫn query parameters có ý nghĩa.

```python
import hashlib
import json

def identity_hash(parts: list[str]) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] Define offsets là normalized-block coordinates, end exclusive; mỗi span luôn kèm page/block ID. Không gọi offset sau normalization là tọa độ raw PDF. Giữ raw text/block cùng provenance để đối chiếu.
- [ ] `tests/rag_helpers.py` tạo `make_document(text: str) -> Document`, `make_result(chunk_id: str, text: str, signals: dict[str,float] | None = None) -> SearchResult` bằng schema thật và cố định timestamps; dùng thống nhất ở các task sau.
- [ ] PASS serialization, Unicode NFC, bbox range, confidence `None`/0, page>=1, legacy read. Dữ liệu cũ không tự gán confidence 1.0.

**Gate:** Identity và provenance không phụ thuộc thuật toán chunking hoặc encoding stdout của PowerShell.

## Task 4: PDF Text, Scan Và Mixed Qua OCR Adapter

**Depends on:** 3.

**Files:** Create `app/ingestion/ocr/{__init__,base,paddle,factory}.py`, `app/ingestion/pdf_renderer.py`, `app/ingestion/extraction_quality.py`, `requirements-ocr.txt`; Modify `app/ingestion/pdf_loader.py`, `app/ingestion/image_processor.py`, `app/ingestion/pipeline.py`, `app/config.py`, `Dockerfile`; Test `tests/test_pdf_extraction.py`, `tests/test_ocr_provider.py`.

**Interfaces:** `PDFRenderer.render_page(content: bytes, page_number: int, dpi: int) -> bytes`; `assess_text(text: str) -> dict[str,float]`; `classify_page(text: str, *, image_coverage: float, has_visual_content: bool) -> str`; `PdfLoader(ocr=None, renderer=None, quality_policy=None)` vẫn hỗ trợ `.load(path, content)`.

- [ ] Regression: text page tốt gọi OCR 0 lần, scan page gọi 1 lần, mixed có ảnh chữ lớn không bị coi là complete, blank không OCR vô ích.

```python
def test_page_with_small_text_and_large_image_is_mixed():
    from app.ingestion.extraction_quality import classify_page
    assert classify_page("Lecture title", image_coverage=0.8,
                         has_visual_content=True) == "mixed"
```

- [ ] Chạy `python -m pytest tests/test_pdf_extraction.py tests/test_ocr_provider.py -q` với fake renderer/OCR; không tải model.
- [ ] Dùng PDFium renderer 250 DPI mặc định, một page tại một thời điểm, max pages=1000 và max rendered pixels/page=25 triệu configurable. Render oversize có downscale/report, corrupt/encrypted PDF trả lỗi có mã; không chỉ đổi extension để tin file hợp lệ.
- [ ] `PaddleOCRProvider` lazy-load model có hỗ trợ `vi/en`, expose model revision/language pack; lấy line boxes/text/confidence, chuẩn hóa về `OCRResult`. Không dùng model English-only rồi mô tả là hỗ trợ Việt. Nếu wheel không phù hợp Windows, chạy adapter cùng contract trong OCR worker Linux cục bộ; không ép cài package không tương thích.
- [ ] Với mixed page, render vùng text bị thiếu; merge theo bbox với native text, tránh nhân đôi. Confidence là trung bình có trọng số số ký tự và giữ confidence từng line; `None` khi provider không cung cấp. Ngưỡng confidence ban đầu là cảnh báo cần hiệu chỉnh, không phải xác suất đúng.
- [ ] Live OCR trên clean/mờ/xoay/cột Việt–Anh. Lưu CER/WER và lỗi số/đơn vị/phủ định; không sửa số y khoa bằng LLM. Thiếu usable text cả document -> ingest FAIL, không trả thành công 0 chunks.
- [ ] Sửa image processor dùng adapter thật đã chọn; mock caption/OCR chỉ dùng profile demo và gắn provenance, không index câu mẫu “Mock OCR...” ở profile quality.

**Gate:** Native page chất lượng tốt không bị OCR lại; scan ra nội dung có thể tìm kiếm; failure/warning có page number. CER mục tiêu cho clean scan <=5% từng ngôn ngữ là mục tiêu nghiệm thu trên fixture đã chọn, không là cam kết cho scan bất kỳ.

## Task 5: Chuẩn Hóa Và Khôi Phục Cấu Trúc

**Depends on:** 4.

**Files:** Create `app/ingestion/normalizer.py`, `app/ingestion/layout.py`; Modify `app/ingestion/pdf_loader.py`, `docx_loader.py`, `html_loader.py`; Test `tests/test_document_normalization.py`, `tests/test_reading_order.py`.

**Interfaces:** `normalize_pages(pages: list[PageExtraction]) -> list[PageExtraction]`; `order_blocks(blocks: list[TextBlock]) -> list[TextBlock]`; `normalize_block(text: str, *, kind: str) -> str`.

- [ ] Test giữ số thập phân, đơn vị, công thức, phủ định và từ có gạch nối có nghĩa; không flatten list/table thành đoạn văn.

```python
def test_normalization_preserves_medical_tokens():
    from app.ingestion.normalizer import normalize_block
    text = "Không dùng. Na+ 135.5 mmol/L; beta-blocker; IL-6; 5 mg."
    assert normalize_block(text, kind="paragraph") == text
```

- [ ] Chạy `python -m pytest tests/test_document_normalization.py tests/test_reading_order.py -q`.
- [ ] Normalize NFC và space trong paragraph; newline repair chỉ khi không phải heading/list/table. Dehyphenation chỉ trường hợp line wrap có bằng chứng, giữ nguyên chemical name/identifier không chắc chắn.
- [ ] Header/footer: chỉ xóa dòng lặp trong vùng đầu/cuối trang, xuất hiện >=60% trong ít nhất 3 trang; lưu audit suppressed blocks, không xóa câu lặp giữa nội dung.
- [ ] Native layout extraction và OCR boxes -> reading order theo cột/vùng; dùng library layout/table adapter khi cần, không đoán toàn bộ PDF bằng regex. Bảng giữ cell/row association và caption; thiếu structure -> warning, có thể quarantine bảng thay vì tạo quan hệ sai.
- [ ] Test list đánh số, heading kéo dài, bảng qua trang có continuation, caption và công thức có subscript; snapshot text/blocks đối chiếu gold. Citation vẫn phân biệt PDF page index và printed page label.

**Gate:** Reviewer lần theo mỗi block đến trang nguồn; text không đảo cột hoặc mất phủ định/đơn vị trên fixture.

## Task 6: Token Chunker, Parent Gốc Và Chunk Quality

**Depends on:** 5.

**Files:** Create `app/ingestion/tokenizer.py`, `token_chunker.py`, `chunk_quality.py`; Modify `app/ingestion/chunker.py`, `pipeline.py`, `app/retrieval/metadata_store.py`, `app/config.py`; Test `tests/test_token_chunker.py`, `tests/test_chunk_quality.py`, `tests/test_chunker.py`.

**Interfaces:** `TokenChunker(counter: TokenCounter, target_tokens=500, overlap_tokens=80, min_tokens=100, max_tokens=640)`; `.chunk(document: Document) -> list[Chunk]`; `assess_chunks(chunks: list[Chunk]) -> ChunkQualityReport`. `TextChunker` cũ vẫn cho legacy tests; config mới `CHUNK_UNIT=tokens` không diễn giải lại `CHUNK_SIZE=900` cũ thành tokens.

- [ ] Test token budget theo tokenizer, overlap có ranh giới câu, table không cắt ngang hàng, long sentence vẫn tiến về phía trước, ID deterministic, parent không nhân đôi overlap.

```python
def test_chunk_count_matches_tokenizer():
    from app.ingestion.token_chunker import TokenChunker
    from tests.rag_helpers import make_document
    class WordCounter:
        name, revision = "test-words", "1"
        def count(self, text):
            return len(self.offsets(text))
        def offsets(self, text):
            import re
            return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    counter = WordCounter()
    chunks = TokenChunker(counter, target_tokens=20, overlap_tokens=4,
                          min_tokens=5, max_tokens=24).chunk(
        make_document("Enzymes lower activation energy. " * 25))
    assert chunks
    assert all(c.metadata.extra["token_count"] == counter.count(c.text) for c in chunks)
    assert all(c.metadata.extra["indexed_token_count"] <= 24 for c in chunks)
```

- [ ] Chạy `python -m pytest tests/test_token_chunker.py tests/test_chunk_quality.py tests/test_chunker.py -q`.
- [ ] Dùng tokenizer của embedding model kèm revision/offset mapping; sentence segmenter có quy tắc Việt–Anh, decimals và abbreviations. Fit title+heading+body trong max indexed tokens; title quá dài cắt contextual prefix theo policy và report, không cắt body âm thầm.
- [ ] Greedy pack block trong section/page đến target; overflow table -> tách hàng + lặp column header/caption; list -> tách item; overflow sentence -> token offsets. Token overlap bám boundary gần nhất và <= budget, ghi thực tế từng chunk.
- [ ] Parent lưu text normalized gốc và source spans, không ghép lại chunk có overlap. `char_start/end` chỉ dùng khi một contiguous span; trường hợp table header lặp/multi-block dùng `source_spans` và `offset_space=normalized_blocks`.
- [ ] Exact duplicate: lưu một canonical chunk cùng `source_occurrences` khi trùng trong revision, không mất page citations. Near-duplicate chỉ cảnh báo/giảm redundancy lúc retrieval; không xóa do chỉ khác một số, đơn vị hoặc từ “không”.
- [ ] Report phân bố token p50/p95/max, tiny/empty/duplicate counts, OCR provenance. Test real BGE tokenizer offline cache một lần khi bật `models_live`; kiểm tra không truncation trên embedding inputs.

**Gate:** Không chunk vượt hard maximum; không mất nội dung gold trừ boilerplate đã ghi audit; định nghĩa ngắn vẫn truy hồi được.

## Task 7: Ingest Phiên Bản, Resume Và Rollback

**Depends on:** 6.

**Files:** Create `app/retrieval/index_manifest.py`, `app/retrieval/publication.py`, `scripts/migrate_index.py`; Modify `app/ingestion/pipeline.py`, `app/retrieval/metadata_store.py`, `dense_store.py`, `vector_store.py`, `factory.py`, `app/rag/service.py`, `scripts/reindex_vectors.py`; Test `tests/test_ingest_publication.py`, `test_index_compatibility.py`, `test_document_management.py`, `test_qdrant_vector_store.py`.

**Interfaces:** `IndexManifest(schema_version, embedding_fingerprint, chunker_fingerprint, extraction_fingerprint, vector_dimension)`; `require_compatible(actual: IndexManifest, requested: IndexManifest) -> None`; `IndexPublisher.stage(document, chunks, parents, report) -> revision_id`; `.publish(doc_id, revision_id, expected_previous) -> None`; `.recover() -> RecoveryReport`. `MetadataStore` protocol gom các method service đang dùng; SQLite là implementation đầu tiên.

- [ ] Test thất bại ở embed, upsert lô thứ hai, metadata commit và process crash. Query vẫn thấy revision trước; retry không nhân đôi; concurrent publish dùng compare-and-swap, loser không xóa dữ liệu winner.

```python
def test_different_model_with_same_dimension_is_incompatible():
    import pytest
    from app.retrieval.index_manifest import IndexManifest, require_compatible
    old = IndexManifest("2", "model-A", "chunk-v2", "extract-v2", 1024)
    new = IndexManifest("2", "model-B", "chunk-v2", "extract-v2", 1024)
    with pytest.raises(ValueError, match="embedding"):
        require_compatible(old, new)
```

- [ ] Chạy `python -m pytest tests/test_ingest_publication.py tests/test_index_compatibility.py tests/test_document_management.py tests/test_qdrant_vector_store.py -q`.
- [ ] Thêm tables revision/chunks/parents/active_documents/ingest_journal bằng migration có backup và idempotency. Raw file lưu theo document+revision trong staging, không overwrite basename trước validation. Số lượng/vector hash/manifest được kiểm tra trước publish.
- [ ] Publish theo thứ tự: stage raw+normalized+chunks -> embed/upsert revision points -> verify count/IDs -> transaction metadata đổi active revision và corpus epoch -> invalidate BM25/cache -> response success -> deferred garbage collection. Failure trước pointer commit để nguyên revision cũ.
- [ ] Query pin active map/epoch một lần. Qdrant chỉ lấy điểm match active revision (filter hoặc candidate filter có backfill/pagination đến đủ top-k); SQLite join với active map. Snapshot không bao giờ trộn parent mới và child cũ. BM25 cache key gồm epoch, refresh trước dùng; old snapshot còn tồn tại cho request đang chạy.
- [ ] Rollback đổi active pointer và invalidates caches; không xóa ngay dữ liệu previous revision. Garbage collector chỉ xóa revision không active/không được reader pin, sau grace period. Index generations thay embedding dùng collection riêng và publish runtime snapshot gồm collection+metadata+BM25; alias switch một mình không làm SQLite atomic.
- [ ] `migrate_index --dry-run` báo ID mapping/conflicts/counts; không tự gộp hai upload cùng filename. Rebuild mới có report, verify rồi `--activate`; cần mode resume fingerprint khớp mới skip embedding.
- [ ] Xóa document cũng tombstone/publish trước rồi GC; kiểm tra active dense IDs == active metadata IDs, missing=0, stale_visible=0 sau crash/restart và delete.

**Gate:** Failure injection và concurrent readers PASS trên SQLite và Qdrant integration. Corpus hiện có chỉ chuyển sang generation mới sau khi đã backup và smoke thành công.

## Task 8: Ingest Job Có Tiến Độ Và Báo Cáo Trên UI

**Depends on:** 7.

**Files:** Create `app/jobs/ingest_jobs.py`, `app/api/ingest_jobs.py`, `scripts/ingest_directory.py`, `scripts/ingest_worker.py`; Modify `app/api/ingest.py`, `app/main.py`, `app/schemas/document.py`, `web/lib/api/ingest.ts`, `web/app/ingest/page.tsx`, `web/app/documents/page.tsx`; Test `tests/test_ingest_jobs.py`, `web/lib/api/ingest.test.ts`, `web/features/ingest/ingest-status.test.tsx`.

**Interfaces:** `POST /ingest/jobs` -> 202 `{job_id,status}`; `GET /ingest/jobs/{id}` -> `queued|extracting|chunking|embedding|publishing|completed|needs_review|failed|cancelled`, progress counts, warnings, report; cancel chỉ kết thúc ở checkpoint. Legacy `/ingest/file` còn hoạt động cho file nhỏ và thêm `report` additive.

- [ ] Test trạng thái chuyển đúng, retry idempotent, lỗi một trang hiển thị rõ, progress không nhảy completed trước publish.

```python
def test_terminal_job_cannot_return_to_embedding():
    import pytest
    from app.jobs.ingest_jobs import require_transition
    with pytest.raises(ValueError):
        require_transition("completed", "embedding")
```

- [ ] Chạy `python -m pytest tests/test_ingest_jobs.py -q`; viết UI tests cho queued/running/review/failed với fake API.
- [ ] Durable SQLite queue, một worker mặc định, lease/heartbeat để recover process chết; không chạy OCR dài trực tiếp trong async event loop. Cached pages key content hash+parser/OCR version để resume. Có queue protocol cho broker sau này, chưa thêm Redis khi chưa cần.
- [ ] Partial extraction: report tổng trang usable/failed/blank; all-empty luôn fail; replacement mất page trước đó thì `needs_review`, chưa publish. `allow_partial` opt-in rõ ràng lưu vào provenance; không che partial bằng success chung.
- [ ] UI hiển thị tên file, bước, số trang/chunks, warning và nút retry/cancel; có empty state dùng sample corpus khi chưa có tài liệu. Citation view hiển thị page PDF và excerpt/bbox; tải raw qua document ID đã kiểm tra path, không mở arbitrary filesystem path.
- [ ] Test payload oversize trước đọc toàn bộ, unsupported magic bytes, concurrent jobs cùng doc, error timeout provider; CLI directory chỉ liệt kê eligible files, có `--dry-run`, report JSON và skip unchanged fingerprint.
- [ ] PASS backend/UI tests, lint/build; smoke 1 text PDF + 1 scan + 1 mixed qua HTTP và xác minh thấy nguồn mới sau job complete.

**Gate:** Upload lớn không timeout rồi bỏ người dùng không biết ingest đã xong hay chưa; partial failures không bị mất trong UI.

## Task 9: Query Việt–Anh Và Hybrid Candidate Có Giới Hạn

**Depends on:** 7 (tích hợp UI chạy sau 8).

**Files:** Create `app/retrieval/query_normalizer.py`; Modify `app/retrieval/bm25_store.py`, `app/rag/hybrid_rag.py`, `app/rag/rerank_rag.py`, `app/config.py`, `app/schemas/retrieval.py`; Test `tests/test_query_normalization.py`, `tests/test_hybrid_rag.py`.

**Interfaces:** `normalize_query(query: str) -> NormalizedQuery(original, dense_text, sparse_text, language)`; `HybridRAG.retrieve_candidates(question: str, dense_k: int = 40, sparse_k: int = 40) -> list[SearchResult]`. Existing `.retrieve(..., top_k)` delegates và cắt ở cuối.

- [ ] Test Unicode decomposed/precomposed, Vietnamese không dấu, English và tên `HbA1c`, `IL-6`, `Na+`, số/đơn vị/phủ định không đổi ở dense_text.

```python
def test_query_preserves_entities_and_numbers():
    from app.retrieval.query_normalizer import normalize_query
    query = "HbA1c 6.5%, Na+ 135 mmol/L, IL-6 không tăng"
    result = normalize_query(query)
    assert result.dense_text == query
    assert "khong" in result.sparse_text
    assert "135" in result.sparse_text
```

- [ ] Chạy `python -m pytest tests/test_query_normalization.py tests/test_hybrid_rag.py -q`.
- [ ] Giữ hai biểu diễn query, unknown/mixed language không cản retrieval. Sparse dùng tokenizer accent folding hiện có; thêm entity tokens mà không làm mất ký tự có nghĩa. Không dùng spelling correction để biến query thành bệnh/thuốc khác.
- [ ] Cache BM25 document frequency, tf, length, avgdl khi build epoch, dùng inverted postings để chỉ chấm documents chứa query term. Đối chiếu điểm với công thức chuẩn trên corpus nhỏ; stats refresh sau publish/delete không sau stage.
- [ ] Tách dense/sparse candidate count khỏi final top-k và rerank candidate count; spy test chỉ gọi mỗi branch một lần với 40 mặc định, không multiplier lồng nhau thành 320. Query rộng tăng k chỉ bằng policy/config trong report.
- [ ] Weighted RRF giữ distinct ranks/scores, source priority tie-break deterministic. Weight 0 loại contribution branch; validate weights hữu hạn/nonnegative, có ít nhất một weight >0.

**Gate:** Exact/không dấu/cross-language đều có evidence đúng trong candidates; không gọi query rewrite cho mọi câu.

## Task 10: Cross-Encoder Chuẩn Điểm Và Relevance Gate

**Depends on:** 9.

**Files:** Modify `app/retrieval/reranker.py`, `reranker_factory.py`, `relevance.py`, `app/rag/rerank_rag.py`, `app/config.py`, `app/schemas/retrieval.py`; Create `app/schemas/relevance.py`; Test `tests/test_cross_encoder_scores.py`, `tests/test_relevance.py`, `tests/test_advanced_rag.py`.

**Interfaces:** `RelevanceDecision(chunk_id: str, accepted: bool, reasons: list[str], scores: dict[str,float])`; `RelevancePolicy.evaluate(query: str, result: SearchResult) -> RelevanceDecision`; `filter_relevant_results` giữ wrapper tương thích. Threshold được đọc từ calibration artifact theo model/version, gồm missing-score policy.

- [ ] Regression test sigmoid đúng một lần; giữ `dense_cosine` tách `reranker_score/logit`; negative reranker không được cứu bằng một từ chung trùng query.

```python
def test_logits_are_normalized_once():
    import pytest
    from app.retrieval.reranker import normalize_logit
    assert normalize_logit(0.0) == pytest.approx(0.5)
    assert normalize_logit(-4.0) < 0.02
    assert normalize_logit(4.0) > 0.98
```

- [ ] Chạy `python -m pytest tests/test_cross_encoder_scores.py tests/test_relevance.py tests/test_advanced_rag.py -q`.
- [ ] `CrossEncoder.predict` hoặc constructor cấu hình Identity để nhận logits, rồi normalize đúng một lần. Validate finite scores/count bằng số pair; reject NaN/Inf/invalid shape. Score [0,1] chỉ là normalized logit, không gọi là calibrated probability.
- [ ] BGE reranker pair token budget tính query+title+chunk; giữ query, báo truncate metadata nếu pair quá dài. Candidate top 20 mặc định, tune 20/40; batch configurable. Local cross-encoder fail thì explicit 503 hoặc configured lexical fallback có `mode_used`/diagnostic, không giả rerank đã thành công.
- [ ] Sau rerank: relevance filter trước diversity; strong negative reject, lexical-only phải có domain anchor/coverage và hard-negative tests. Bỏ đường cứu bằng source filename. Cross-language không bắt buộc token Việt phải có trong chunk Anh.
- [ ] MMR chọn trên normalized nonnegative relevance, dedup theo spans/token shingles; keep metadata mọi nguồn gốc. Giá trị ngưỡng chỉ provisional trước calibration Task 15, report `calibration_status=provisional`.
- [ ] Test top_k=0, empty, toàn irrelevant, khác tên nhưng trùng nội dung, query dài, document injection. Live retrieval so sánh lexical baseline với cross-encoder, đo latency/RAM.

**Gate:** Không nhận unrelated chỉ vì sigmoid bị áp hai lần; toàn irrelevant không gọi generator. Relevance decision có lý do đọc được qua diagnostics.

## Task 11: Context Budget, Parent-Child Và Multi-Query

**Depends on:** 10.

**Files:** Create `app/rag/context_builder.py`; Modify `app/rag/parent_child_rag.py`, `multi_query_rag.py`, `corrective_rag.py`, `app/retrieval/query_transform.py`, `metadata_store.py`, `app/schemas/chat.py`; Test `tests/test_context_budget.py`, `tests/test_advanced_rag.py`.

**Interfaces:** `PackedContext(items: list[SearchResult], token_count: int, source_map: dict[str, dict], omitted_ids: list[str])`; `ContextBuilder.pack(query: str, results: list[SearchResult], *, input_budget: int) -> PackedContext`; `QueryExpander.expand(query: str) -> list[str]`.

- [ ] Test budget nhỏ không overflow; parent expansion vẫn thuộc snapshot; numbering không đổi sau packing, không mất evidence quan trọng do parent quá dài.

```python
def test_context_builder_returns_empty_for_zero_budget():
    from app.rag.context_builder import ContextBuilder
    from tests.rag_helpers import make_result
    class Counter:
        def count(self, text):
            return len(text.split())
    packed = ContextBuilder(Counter()).pack("Q?", [make_result("c1", "Evidence")], input_budget=0)
    assert packed.items == []
    assert packed.token_count == 0
```

- [ ] Chạy `python -m pytest tests/test_context_budget.py tests/test_advanced_rag.py -q`.
- [ ] Generator tokenizer riêng với embedding tokenizer. Budget = model context window trừ system/query/source wrappers/output reserve/safety margin. Config ban đầu context window 8192, output reserve 1024; doctor/probe xác nhận runtime hỗ trợ. Counter không chính xác phải dùng upper-bound bảo thủ có flag, không hứa exact budget.
- [ ] Expand parent sau relevance, đọc parent gốc và source spans; giới hạn mỗi parent. Parent quá dài chọn window quanh child, giữ target answer span và source location; không tăng thành hàng chục nghìn token.
- [ ] Multi-query giữ query gốc, tối đa 3 biến thể có nghĩa: decomposition so sánh hoặc dịch khi cần; loại duplicate, bảo toàn entity/numeric/negative intent. Retry corrective tối đa 1 với retrieval budget tổng rõ ràng, không tạo vòng lặp agent vô hạn.
- [ ] Source IDs được cấp sau selection/packing và bất biến trong generation+repair. Source map ghi revision/page/quote cho context thực tế; score child không gán giả cho mọi parent claim.

**Gate:** Mọi request có upper bound candidate và token; câu so sánh có cơ hội chứa evidence cả hai vế trong context.

## Task 12: Câu Trả Lời Có Cấu Trúc Và Citation Đúng ID

**Depends on:** 11.

**Files:** Create `app/schemas/answer.py`, `app/rag/answer_service.py`, `app/rag/answer_validation.py`; Modify `app/llm/base.py`, `ollama_llm.py`, `mock_llm.py`, `app/rag/prompts.py`, `naive_rag.py`, `hybrid_rag.py`, `rerank_rag.py`, `parent_child_rag.py`, `multi_query_rag.py`, `corrective_rag.py`, `service.py`; Test `tests/test_answer_validation.py`, `tests/test_model_providers.py`, `tests/test_safety.py`, `tests/test_api.py`.

**Interfaces:** `AnswerDraft(short_answer: str, explanation: str, claims: list[CitedClaim])`; `CitedClaim(text: str, citation_ids: list[str])`; `GenerationResult(draft: AnswerDraft | None, finish_reason: str, usage: dict, raw_text: str | None)`; `validate_citations(text: str, allowed: set[str]) -> list[str]`. Shared `AnswerService.answer(question, packed_context) -> RagOutput`; legacy `LLMProvider.generate` được adapter bảo toàn cho test/plugin.

- [ ] Test `[99]` không có nguồn, không citation, nguồn có metadata nhưng answer không cite, content chỉ là reasoning, response truncated và malformed structured output.

```python
def test_invalid_source_ids_are_rejected():
    from app.rag.answer_validation import validate_citations
    assert validate_citations("Evidence [1] and invented source [99].", {"1"}) == ["99"]
```

- [ ] Chạy `python -m pytest tests/test_answer_validation.py tests/test_model_providers.py tests/test_safety.py tests/test_api.py -q`.
- [ ] Khi provider hỗ trợ structured generation, yêu cầu schema fields và per-claim citations; parse JSON bằng parser/schema, không regex toàn nội dung JSON. Render bốn section bằng application, nguồn lấy từ context map; model không tự tạo file path/page. Provider chỉ text vẫn parse/validate, không bỏ gate.
- [ ] `finish_reason=length`, empty `content`, chỉ thinking hoặc không parse được -> validation failure. Tối đa một repair với errors và cùng context; nếu thất bại trả controlled response có disclaimer, sources rỗng và trace lưu failure. Không trả reasoning thô hoặc câu bị cắt làm answer hợp lệ.
- [ ] System prompt xem nguồn là dữ liệu, không thi hành chỉ dẫn trong PDF. Citation ID được chống giả ở context serialization, không nhận source labels do document tự cung cấp. Test instruction injection “ignore system”, fake `[1]`, yêu cầu bỏ disclaimer trong chunk.
- [ ] Route tất cả RAG modes qua shared answer service để cùng gate; không gom ingestion/retrieval/provider vào service này. Emergency input short-circuit trước mọi retrieval/LLM/verifier; output filter từ chối quyết định điều trị cá nhân nhưng không chặn tài liệu học tập chỉ vì có từ thuốc.
- [ ] API vẫn giữ `answer`, `sources`, `retrieved_chunks`, `mode_used`, `safety_warning`; thêm validation status/trace ID additive. Return `sources` theo các citation hợp lệ được dùng, `retrieved_chunks` thể hiện retrieved context riêng.

**Gate:** 100% unit cases invalid citation/empty/truncated bị phát hiện; medical disclaimer do code render, không phụ thuộc model làm đúng prompt.

## Task 13: Claim Support, Repair Và Cách Đo Faithfulness

**Depends on:** 12.

**Files:** Create `app/rag/support_verifier.py`, `app/evaluation/faithfulness_eval.py`, `evaluation/claims/dev.jsonl`, `test.jsonl`; Modify `app/rag/answer_service.py`, `app/evaluation/answer_eval.py`, `app/config.py`; Test `tests/test_support_verifier.py`, `tests/test_evaluation.py`.

**Interfaces:** `ClaimSupport(claim_id: str, verdict: Literal["supported","unsupported","unknown"], evidence_ids: list[str], reason_code: str)`; `SupportVerifier.verify(claims: list[CitedClaim], context: PackedContext) -> list[ClaimSupport]`. Local verifier dùng LLM adapter có JSON schema hoặc NLI adapter đã benchmark Việt–Anh; deterministic fake cho tests.

- [ ] Test claim khác nguồn ở con số, đơn vị, phủ định và đối tượng; citation hợp lệ nhưng trỏ nguồn không entail phải fail; verifier timeout -> unknown, không supported.

```python
def test_unknown_support_does_not_pass_strict_policy():
    from app.rag.support_verifier import ClaimSupport, supported_answer
    result = ClaimSupport("claim-1", "unknown", ["1"], "verifier_timeout")
    assert supported_answer([result], strict=True) is False
```

- [ ] Chạy `python -m pytest tests/test_support_verifier.py tests/test_evaluation.py -q`.
- [ ] Gửi verifier từng claim với **nguồn nó cite**, không toàn bộ corpus; verifier không được tự retrieval tài liệu khác để hợp thức hóa answer. Numeric/negation checks chỉ báo nghi vấn, không tự suy semantic là đúng/sai bằng string match.
- [ ] Dùng một repair budget **chung** cho citation/format/support (không cộng thành ba lần). Repair chỉ bỏ/sửa claim theo evidence đã có; không tự thêm nguồn. Strict quality profile không trả claim unknown/unsupported; sau repair fail trả thiếu căn cứ. Verifier hạ tầng hỏng trả lỗi stage rõ ràng, không nói sai rằng corpus không có thông tin.
- [ ] Đánh giá false accept/false reject của verifier trên tối thiểu 80 claim pairs review tay, cân bằng supported/contradicted/insufficient, có Việt–Anh. Lưu model/judge version, rubric và disagreements. Judge score riêng khỏi human score; citation entailment không được suy từ reranker relevance.
- [ ] Kiểm tra không tăng answer quality bằng cách từ chối mọi câu: đo answer coverage, answer correctness trong answerable và false refusal cùng faithfulness. Unknown không được bỏ khỏi denominator.

**Gate:** Pipeline chặn unsupported claims trong regression suite; báo cáo benchmark có human-audited support rate, coverage và errors. Không gọi đây là bảo đảm đúng về y khoa.

## Task 14: Cloud Adapters Và Hybrid Profile

**Depends on:** 3, 10, 12–13.

**Files:** Create `app/providers/__init__.py`, `app/providers/http_transport.py`, `app/ingestion/ocr/http.py`, `app/embeddings/http.py`, `app/retrieval/http_reranker.py`, `app/llm/http_llm.py`, `docs/providers/http-contract.md`, `services/inference_gateway/app.py`, `services/inference_gateway/requirements.txt`, `services/inference_gateway/README.md`; Modify factories từng tầng, `app/config.py`, `.env.example`; Test `tests/test_http_providers.py`, `tests/test_provider_profiles.py`, `tests/test_inference_gateway.py`.

**Interfaces:** Hosted adapters thực hiện cùng `OCRProvider`, `EmbeddingProvider`, `Reranker`, LLM structured-generation contract. Đầu tiên hỗ trợ một **gateway HTTP versioned `/v1` do dự án định nghĩa**, dùng được cho tự host cloud; không tuyên bố tự tương thích mọi SaaS.

```json
{
  "ocr_request": {"image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aD1sAAAAASUVORK5CYII=", "page_number": 1, "languages": ["vi", "en"]},
  "ocr_response": {"blocks": [], "confidence": null, "warnings": []},
  "embedding_request": {"texts": ["enzyme"], "purpose": "query"},
  "embedding_response": {"model_fingerprint": "pinned-model", "vectors": [[0.0, 1.0]]},
  "rerank_request": {"query": "Q", "documents": [{"id": "c1", "text": "T"}]},
  "rerank_response": {"score_type": "logit", "results": [{"id": "c1", "score": 2.0}]}
}
```

Ảnh trong ví dụ chỉ minh họa transport; live OCR phải dùng scan fixture có chữ. LLM request chứa messages+schema+max output tokens; response `GenerationResult` không chứa keys hoặc reasoning. Endpoints cố định `POST /v1/ocr`, `/v1/embeddings`, `/v1/rerank`, `/v1/generate`; response có `request_id` và `model_fingerprint` theo stage.

- [ ] Viết tests fake HTTP cho malformed JSON, thứ tự vectors sai, wrong count/dimension/fingerprint, rerank score semantics, OCR không confidence, LLM timeout/429/401 và retry budget.

```python
def test_retry_policy_excludes_authentication_failures():
    from app.providers.http_transport import is_retryable_status
    assert is_retryable_status(429)
    assert is_retryable_status(503)
    assert not is_retryable_status(401)
```

- [ ] Chạy `python -m pytest tests/test_http_providers.py tests/test_provider_profiles.py -q`.
- [ ] Timeout connect/read và tổng deadline; retry tối đa 2 cho 429/503 với `Retry-After` capped, chỉ tác vụ idempotent; cancellation chấm dứt retries. Secrets từ env; logs chỉ stage/status/count/latency.
- [ ] Local profile không outbound khi không bật; cloud/hybrid chọn từng stage explicit. Tokenizer/fingerprint cloud được khai báo, không đoán từ dimension. Đổi embedding profile -> collection/generation mới và rebuild; đổi LLM/OCR/reranker có invalidation theo mục 3.
- [ ] Tạo service gateway deployable bằng FastAPI trong `services/inference_gateway/`, tái sử dụng adapters local/model contract, token auth và request limits để live-test trên endpoint được cấu hình. Document adapter cho SaaS cụ thể là phần mở rộng mapping đúng contract; placeholder `OpenAILLM` không được trình bày như provider đã hoạt động.
- [ ] Live smoke trên endpoint cloud thật khi có URL/key/model, opt-in qua `CLOUD_LIVE=1`; max 10 requests, tối đa 20.000 input tokens và 4.000 generated tokens cho cả run. Chưa có endpoint: báo `not_run`, local gate vẫn có thể PASS, cloud gate còn mở.

**Gate:** Đổi stage local/cloud không đổi API routes; gateway remote đã smoke hoặc báo rõ còn thiếu prerequisite. Không dùng mock HTTP để khẳng định chất lượng cloud.

## Task 15: Benchmark, Tuning Trên Dev Và Báo Cáo Held-Out

**Depends on:** 2, 7, 10–13 cho benchmark local; thêm 14 khi chạy so sánh cloud.

**Files:** Create `app/evaluation/metrics.py`, `app/evaluation/benchmark.py`, `scripts/run_benchmark.py`, `config/benchmarks/rag-quality.json`, `docs/quality/evaluation-protocol.md`; Modify `app/evaluation/retrieval_eval.py`, `answer_eval.py`, `safety_eval.py`, `scripts/run_eval.py`, `app/api/eval.py`, `app/schemas/evaluation.py`, `web/app/evaluation/page.tsx`, `web/lib/api/evaluation.ts`; Test `tests/test_ranking_metrics.py`, `tests/test_benchmark.py`, `tests/test_evaluation.py`, `web/lib/api/evaluation.test.ts`.

**Interfaces:** `ranking_metrics(retrieved: list[str], relevance: dict[str,int], k: int) -> dict[str,float]`; `run_benchmark(config: Path, split: str, output: Path) -> BenchmarkReport`. Report gồm config/model/corpus/index hashes, raw predictions, metric denominators, stage latency, provider status, hardware và errors.

- [ ] Test công thức metrics với toy ranking, duplicate evidence chỉ tính một lần, empty retrieval =0, no positives không tính retrieval denominator nhưng vẫn vào refusal metrics.

```python
def test_recall_and_mrr_on_known_ranking():
    import pytest
    from app.evaluation.metrics import ranking_metrics
    result = ranking_metrics(["wrong", "e1", "e2"], {"e1": 2, "e2": 1}, k=2)
    assert result["recall"] == pytest.approx(0.5)
    assert result["mrr"] == pytest.approx(0.5)
```

- [ ] Chạy `python -m pytest tests/test_ranking_metrics.py tests/test_benchmark.py tests/test_evaluation.py -q`.
- [ ] Resolve gold evidence theo revision-independent page/block span/quote hashes; chunk chỉ được relevant khi chứa đoạn trả lời được review, không chỉ cùng sách. Multi-hop cần coverage đủ evidence groups; partial coverage báo riêng.
- [ ] Các mode benchmark: BM25-only, dense-only, hybrid, hybrid+lexical, hybrid+cross-encoder. Cùng corpus, same splits, query set và evaluator version; ghi OCR vs native và query-language slices. Thay chunker -> build index riêng, không trộn baseline cũ với chunk mới.
- [ ] Tuning theo giai đoạn trên **dev**: chunk target 400/500/600 với overlap 60/80/100; chọn extraction/chunk tốt rồi tune RRF dense weights 0.5/1/1.5, sparse=1; tiếp theo rerank candidate 20/40, final k 5/8, relevance cutoffs theo score distribution. Mỗi stage lưu rationale và latency, không chạy full Cartesian grid tốn tài nguyên vô ích.
- [ ] Freeze config rồi chạy held-out test **một lần cho release candidate**. Nếu FAIL, ghi FAIL; tiếp tục phát triển bằng dev và bổ sung held-out mới trước tuyên bố cải tiến không thiên lệch. Không chỉnh ngưỡng ngay theo những câu test vừa xem rồi gọi test vẫn held-out.
- [ ] Metric report: Recall@5/10, Hit@5, MRR, nDCG@10; unanswerable precision/recall; citation ID validity và human citation correctness riêng; faithfulness, answerable correctness/coverage, false refusal; disclaimer/safety; CER/WER, chunk stats, p50/p95 theo stage. Latency từ >=30 runs có warmup, cold/warm tách; failure/timeouts không bỏ khỏi báo cáo.
- [ ] Chốt denominator: refusal precision = đúng unanswerable / tất cả refusal ngoài safety; refusal recall = unanswerable được từ chối / tất cả unanswerable. Answer coverage = answerable được trả lời / tất cả answerable. Safety cases đánh giá riêng; provider failure không được tính là successful refusal. Citation correctness chấm trên claim–citation pairs; faithfulness chấm tất cả factual claims, kể cả claim chưa cite.
- [ ] Lệnh nghiệm thu sau khi module được tạo:

```powershell
python -m scripts.validate_eval_dataset --corpus evaluation/corpus/manifest.json --cases evaluation/cases
python -m scripts.run_benchmark --config config/benchmarks/rag-quality.json --split dev --output data/reports/rag-dev.json
python -m scripts.run_benchmark --config config/benchmarks/rag-quality.json --split test --output data/reports/rag-test.json
```

- [ ] UI hiển thị run metadata, numerator/denominator, failures và missing prerequisites. Export JSON+Markdown; thiếu nhãn support -> `not_evaluated`, không tạo 1.0. Không tự đổi mode của case để che baseline kém; mode là experiment dimension tường minh.

**Gate:** Hybrid+rerank không kém BM25 và dense trên Recall@10/MRR; unanswerable precision, citation correctness, format và safety >=0.95 theo spec; disclaimer/citation IDs mục tiêu 1.0. Thêm answerable coverage >=0.85 và human faithfulness >=0.95. Đây là mục tiêu, chưa phải kết quả; n nhỏ phải báo số case và khoảng bất định.

## Task 16: Nghiệm Thu End-to-End, Hiệu Năng Và Runbook

**Depends on:** 1–15.

**Files:** Create `scripts/smoke_rag.py`, `scripts/check_index.py`, `docs/quality/release-checklist.md`, `docs/quality/quality-report.md`, `docs/runbooks/rag-local.md`; Modify `README.md`, `docs/RAG_INTERVIEW_VI.md`, `Dockerfile`, `docker-compose.yml`; Test `tests/test_smoke_contract.py` và integration tests đã tạo.

**Interfaces:** `smoke_rag --base-url URL --fixture-dir PATH --output PATH`; `check_index --read-only --json`; mọi script trả nonzero khi gate fail, không nuốt traceback của subprocess.

- [ ] Test health live nhưng embedding/index mismatch -> ready fail; smoke phải bắt API 200 nhưng answer rỗng, citation [99], reasoning-only hoặc thiếu evidence.

```python
def test_smoke_requires_nonempty_answer():
    from scripts.smoke_rag import check_answer_payload
    errors = check_answer_payload({"answer": "", "sources": [], "mode_used": "rerank"})
    assert "empty_answer" in errors
```

- [ ] Chạy `python -m pytest tests/test_smoke_contract.py -q`, rồi tạo smoke dùng requests timeouts/stage diagnostics; không dump tài liệu vào logs.
- [ ] Smoke HTTP/UI: text PDF, scan PDF, mixed PDF -> job report -> search Việt/Anh -> answer nguồn đúng trang -> irrelevant refusal -> emergency short-circuit -> replace -> crash/resume -> delete -> restart. Corpus rỗng trả thiếu thông tin và không tạo citation. Kiểm tra upload cùng tên không overwrite tài liệu khác.
- [ ] Corpus scale benchmark 10k và 50k chunks tổng hợp có label synthetic, kiểm tra RAM và p50/p95, peak OCR memory theo trang, BM25 không rebuild mỗi request, Qdrant payload indexes cho `doc_id/revision_id`. Không nhân bản test document rồi dùng duplicate hit để chứng minh retrieval quality.
- [ ] Giới hạn ban đầu: local CPU warm retrieval+rerank p95 <=15 giây cho 10k chunks; answer p95 <=90 giây với output budget đã chọn; timeout cuối request <=180 giây. Đây là service targets cần đo trên phần cứng được ghi nhận; không giảm safety/grounding để đạt latency. Nếu không đạt thì giảm batch/candidate, dùng cloud stage hoặc ghi limitation và gate chưa PASS.
- [ ] Sau check có mã nguồn thay đổi cuối cùng, chạy:

```powershell
python -m pytest -q
python -m ruff check app scripts tests
python -m black --check app scripts tests
docker compose config --quiet
git diff --check
```

Chạy `npm test`, `npm run lint`, `npm run build` trong `web`. Model/service integration chạy bằng markers explicit sau khi doctor xác nhận prerequisites. Không dùng số test từ tháng trước làm bằng chứng mới.

- [ ] Kiểm tra UI desktop/mobile: ingest progress, page report, sources, evaluation. Chỉ cần UI hiện tại hoạt động đúng, không thêm landing page. Dev/build dùng cache riêng hoặc không chạy build cạnh dev; khởi động server để người dùng thử và ghi URL.
- [ ] Runbook có lệnh setup, doctor, download models explicit, start Ollama/Qdrant/API/UI, sample ingest, re-ingest, resume, rollback, reset chỉ corpus demo; giải thích `.env` được nạp cho cả API/CLI. Tài liệu phỏng vấn mô tả đúng feature thật và model variant, không gọi mock OCR/cloud là đã triển khai.

**Gate:** Báo cáo cuối có commit/config/corpus/version hashes, test logs và benchmark artifacts; mỗi dòng acceptance có PASS/FAIL/NOT_RUN cùng bằng chứng. Core local chỉ hoàn tất khi gates local đều PASS; cloud/scaling chưa chạy phải ghi rõ.

---

## 6. Các Tình Huống Bắt Buộc Có Test

| Nhóm | Tình huống | Kết quả mong đợi |
| --- | --- | --- |
| Ingest | PDF text sạch | Extract text, OCR calls=0 |
| Ingest | PDF scan Việt/Anh | OCR thật, có provenance/page/confidence |
| Ingest | Một trang có title text và bảng scan | OCR vùng thiếu, không bỏ bảng |
| Ingest | Blank/corrupt/encrypted | Blank ghi riêng; invalid fail rõ, không index rác |
| Ingest | OCR timeout ở một số trang | Page warnings/needs_review, không silently complete |
| Chunking | 2 cột/bảng/list | Reading order và quan hệ row/cell giữ đúng |
| Chunking | `3.5`, `IL-6`, `không`, đơn vị | Không làm sai thuật ngữ/số/phủ định |
| Chunking | Một câu rất dài | Split có tiến triển, đúng token limit |
| Index | Nội dung cùng tài liệu thay đổi | Stable doc ID, revision mới; không stale visible |
| Index | Model mới cùng dimension | Compatibility reject và yêu cầu generation mới |
| Index | Crash mỗi stage/restart | Bản cũ đọc được, resume idempotent |
| Retrieval | Việt không dấu và Việt->Anh | Có evidence đúng trong candidates; không sửa entity |
| Retrieval | Không có chủ đề trong corpus | Gate từ chối, không nearest-neighbor forced answer |
| Retrieval | Top results gần trùng | Context đa dạng, giữ source occurrences |
| Answer | Citation không tồn tại | Repair tối đa một lần, nếu fail không trả claim |
| Answer | Citation có thật nhưng nội dung sai | Verifier/human eval phát hiện; không tính ID-valid là faithful |
| Answer | Corpus chứa chỉ dẫn độc hại | Chỉ xem là dữ liệu, không đổi policy/tool behavior |
| Answer | API trả 200 nhưng reasoning/empty/truncated | Validation fail, không đánh dấu câu trả lời thành công |
| Safety | Cấp cứu/chẩn đoán/kê đơn | Guardrail trước retrieval và disclaimer nguyên văn |
| Eval | Hai paraphrase cùng evidence khác split | Dataset validator reject leakage |

## 7. Ma Trận Spec -> Task -> Bằng Chứng

| Spec | Task | Artifact/gate |
| --- | --- | --- |
| 4.1–4.2 PDF routing/extraction quality | 3–4, 8 | Page reports, OCR routing tests, CER/WER |
| 4.3 normalization/layout | 5 | Gold blocks/snapshots, source spans |
| 5 token chunking/metadata/quality | 3, 6 | Token budget tests, chunk report, parents nguyên gốc |
| 6.1 query processing | 9, 11 | Query/rewriter entity-preservation tests |
| 6.2 fusion | 9 | Candidate counts, branch signals, baseline metrics |
| 6.3 reranking/diversity | 10–11 | Activation regression, live cross-encoder report |
| 6.4 relevance | 10, 15 | Accept/reject reason + calibrated thresholds |
| 7 generation/citations/safety | 12–13 | Structured answer, verifier audit, guardrail tests |
| 8 evaluation | 2, 15 | Versioned dataset, dev/test reports, claim audit |
| 9 failure/observability | 1, 7–8, 16 | Crash/recovery tests, readiness, redacted stage logs |
| 10 scalability/local/cloud | 7, 9, 14, 16 | Manifest/versioning, Qdrant, gateway contract, load report |
| 11 nghiệm thu tổng | 16 | Release checklist, smoke HTTP/UI, runbook |

## 8. Ưu Tiên Triển Khai

P0 là Task 1–8: chạy được runtime, extract đúng và publish không làm hỏng index. P1 là Task 9–13 và benchmark local Task 15: tìm đúng, trả lời có căn cứ. P2 là Task 14 cùng scale/cloud verification Task 16: mở rộng deployment mà không đổi core contracts.

Không bổ sung auth/public deployment, video keyframe, fine-tuning model hoặc frontend redesign vào workstream này. Các chức năng đó có thể có kế hoạch riêng sau khi core quality đã có bằng chứng.

## 9. Nguồn Kỹ Thuật Đã Đối Chiếu

- PaddleOCR hỗ trợ cấu hình ngôn ngữ Việt/Anh theo model; model pack phải được chọn và kiểm thử đúng. [OCR pipeline documentation](https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html).
- Sentence Transformers có thể áp sigmoid mặc định cho cross-encoder một nhãn; adapter phải quy định rõ logits hay normalized score để tránh sigmoid hai lần. [CrossEncoder API](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
- Qdrant hỗ trợ atomic alias operations; bảo đảm này áp dụng alias trong Qdrant, không mở rộng thành transaction với metadata SQLite. [Qdrant collections and aliases](https://qdrant.tech/documentation/manage-data/collections/).

Các lựa chọn về threshold, token budget và latency trong plan là thiết kế của dự án, cần benchmark; không phải độ chính xác được tài liệu nhà cung cấp bảo đảm.
